"""Contained obstore payload mechanics for authority-owned generations.

The participant supplies only validated immutable object effects. The existing
lifecycle engine remains responsible for preparation, verification, promotion,
replay, and cleanup debt.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any
from urllib.parse import urlparse

from obstore.exceptions import (
    AlreadyExistsError,
    BaseError as ObstoreError,
    GenericError,
    NotFoundError,
    PreconditionError,
)

from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheConfigurationError,
    CacheReason,
    CacheUnsafePathError,
)
from cacheness.interfaces import GuardedReadSnapshot, GuardedWriteResult

from .guarded_handler_io import GuardedHandlerIO, GuardedStagedArtifact


DEFAULT_MAX_TRANSFER_BYTES = 128 * 1024 * 1024
_LOCATOR_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")


class ObstoreGenerationIO:
    """Bridge guarded path handlers to a single obstore object store."""

    topology_capabilities = {
        "durable": False,
        "process_scope": "process",
        "host_scope": "process",
        "immutable_generations": True,
        "streaming": True,
        "listing": False,
    }

    def __init__(
        self,
        store: object,
        handler_io: GuardedHandlerIO,
        *,
        qualification_identity: str,
        max_upload_bytes: int = DEFAULT_MAX_TRANSFER_BYTES,
        max_download_bytes: int = DEFAULT_MAX_TRANSFER_BYTES,
    ) -> None:
        for method in ("put", "get", "head", "delete"):
            if not callable(getattr(store, method, None)):
                raise TypeError(f"obstore participant requires a store with {method}()")
        if not isinstance(handler_io, GuardedHandlerIO):
            raise TypeError("handler_io must be a GuardedHandlerIO")
        if qualification_identity not in {"filesystem", "memory", "s3"}:
            raise ValueError("qualification_identity must name a known payload topology")
        self._store = store
        self._handler_io = handler_io
        self.qualification_identity = qualification_identity
        self.max_upload_bytes = self._transfer_limit(max_upload_bytes, "max_upload_bytes")
        self.max_download_bytes = self._transfer_limit(
            max_download_bytes, "max_download_bytes"
        )
        self._closed = False
        if qualification_identity == "filesystem":
            self.topology_capabilities = {
                **self.topology_capabilities,
                "durable": True,
                "process_scope": "host",
                "host_scope": "host",
                "listing": True,
            }
        elif qualification_identity == "s3":
            self.topology_capabilities = {
                **self.topology_capabilities,
                "durable": True,
                "process_scope": "multi_host",
                "host_scope": "multi_host",
                "listing": True,
            }

    @classmethod
    def for_s3(
        cls,
        *,
        bucket: str,
        prefix: str,
        region: str,
        handler_root: Path | str,
        endpoint: str | None = None,
        allow_test_endpoint: bool = False,
        max_sdk_retries: int = 1,
        max_upload_bytes: int = DEFAULT_MAX_TRANSFER_BYTES,
        max_download_bytes: int = DEFAULT_MAX_TRANSFER_BYTES,
    ) -> "ObstoreGenerationIO":
        """Build the shared participant for explicitly configured Amazon S3.

        Production uses obstore's standard AWS credential chain.  An endpoint
        override is deliberately limited to the loopback HTTP moto fixture so
        compatible object stores cannot inherit Amazon S3 qualification.
        """

        if not isinstance(bucket, str) or not bucket.strip():
            raise CacheConfigurationError("S3 bucket must be a non-empty string")
        if not isinstance(region, str) or not region.strip():
            raise CacheConfigurationError("S3 region must be a non-empty string")
        if not isinstance(prefix, str) or not prefix.strip("/"):
            raise CacheConfigurationError("S3 prefix must be a non-empty string")
        if type(max_sdk_retries) is not int or not 0 <= max_sdk_retries <= 3:
            raise CacheConfigurationError(
                "S3 max_sdk_retries must be an integer between zero and three"
            )
        if not isinstance(allow_test_endpoint, bool):
            raise CacheConfigurationError("allow_test_endpoint must be a boolean")

        store_kwargs: dict[str, Any] = {
            "prefix": prefix.strip("/"),
            "config": {"region": region.strip(), "conditional_put": "etag"},
            "retry_config": {"max_retries": max_sdk_retries},
        }
        if endpoint is not None:
            if not allow_test_endpoint or not cls._is_loopback_moto_endpoint(endpoint):
                raise CacheConfigurationError(
                    "S3 endpoint overrides are allowed only for an explicit loopback moto test"
                )
            store_kwargs["endpoint"] = endpoint
            store_kwargs["allow_http"] = True
        elif allow_test_endpoint:
            raise CacheConfigurationError(
                "allow_test_endpoint requires an explicit loopback moto endpoint"
            )

        from obstore.store import S3Store

        return cls(
            S3Store(bucket.strip(), **store_kwargs),
            GuardedHandlerIO(handler_root),
            qualification_identity="s3",
            max_upload_bytes=max_upload_bytes,
            max_download_bytes=max_download_bytes,
        )

    @staticmethod
    def _is_loopback_moto_endpoint(endpoint: object) -> bool:
        """Accept only the explicit local HTTP exception used by moto tests."""

        if not isinstance(endpoint, str) or not endpoint:
            return False
        parsed = urlparse(endpoint)
        return (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and parsed.port is not None
        )

    @staticmethod
    def _transfer_limit(value: int, field_name: str) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
        return value

    def materialize_handler_io(self) -> "ObstoreGenerationIO":
        """Return the one guarded generation-I/O object consumed by BlobStore."""
        self._require_open()
        return self

    @property
    def root(self) -> Path:
        """Expose the guarded root only for existing lifecycle path validation."""
        return self._handler_io.root

    @contextmanager
    def stage(self, handler: Any, data: Any, config: Any) -> Iterator[GuardedStagedArtifact]:
        """Delegate all handler-facing staging to the guarded private boundary."""
        self._require_open()
        with self._handler_io.stage(handler, data, config) as staged:
            yield staged

    def publish_generation(
        self,
        staged: GuardedStagedArtifact,
        locator: Path | str,
    ) -> GuardedWriteResult:
        """Create one immutable object from the retained staged descriptor."""
        self._require_open()
        locator_text = self._validated_locator(locator)
        with staged.open() as (source, byte_size):
            if byte_size > self.max_upload_bytes:
                raise CacheBlobBackendError(
                    "Staged generation exceeds configured upload bound",
                    context={"operation": "obstore.publish", "stage": "size"},
                )
            try:
                result = self._store.put(
                    locator_text,
                    source,
                    mode="create",
                    use_multipart=False,
                )
            except (AlreadyExistsError, PreconditionError) as error:
                raise FileExistsError("Immutable generation already exists") from error
            except (OSError, GenericError) as error:
                result = self._settle_ambiguous_publication(
                    staged,
                    locator_text,
                    byte_size,
                    error,
                )
            except (ObstoreError, TypeError, ValueError) as error:
                raise CacheBlobBackendError(
                    "Obstore immutable generation publication failed",
                    context={"operation": "obstore.publish", "stage": "put"},
                ) from error
        metadata = dict(result) if isinstance(result, Mapping) else {}
        published = staged.result_for(Path(locator_text), byte_size)
        published["metadata"] = metadata
        return published

    @contextmanager
    def open_snapshot(
        self,
        locator: Path | str,
        metadata: dict[str, Any],
    ) -> Iterator[GuardedReadSnapshot]:
        """Copy one exact bounded object into a sealed private handler snapshot."""
        self._require_open()
        locator_text = self._validated_locator(locator)
        try:
            result = self._store.get(locator_text)
        except (NotFoundError, FileNotFoundError) as error:
            raise FileNotFoundError("Obstore generation is absent") from error
        except (ObstoreError, OSError, TypeError, ValueError) as error:
            raise CacheBlobBackendError(
                "Obstore generation read request failed",
                context={"operation": "obstore.snapshot", "stage": "get"},
            ) from error

        result_metadata = getattr(result, "meta", None)
        if not isinstance(result_metadata, Mapping) or type(result_metadata.get("size")) is not int:
            raise CacheBlobBackendError(
                "Obstore generation metadata lacks a valid byte size",
                context={"operation": "obstore.snapshot", "stage": "head"},
            )
        expected_size = result_metadata["size"]
        if expected_size < 0 or expected_size > self.max_download_bytes:
            raise CacheBlobBackendError(
                "Obstore generation exceeds configured download bound",
                context={"operation": "obstore.snapshot", "stage": "size"},
            )

        observed = 0
        try:
            chunks = result.stream(min_chunk_size=1)
            with self._handler_io.open_snapshot_sink(locator_text, metadata) as sink:
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise CacheBlobBackendError(
                            "Obstore generation stream yielded non-bytes",
                            context={"operation": "obstore.snapshot", "stage": "stream"},
                        )
                    observed += len(chunk)
                    if observed > expected_size or observed > self.max_download_bytes:
                        raise CacheBlobBackendError(
                            "Obstore generation stream exceeded its declared bound",
                            context={"operation": "obstore.snapshot", "stage": "stream"},
                        )
                    sink.destination.write(chunk)
                if observed != expected_size:
                    raise CacheBlobBackendError(
                        "Obstore generation stream disagrees with exact object size",
                        context={"operation": "obstore.snapshot", "stage": "stream"},
                    )
                yield sink.finish()
        except CacheBlobBackendError:
            raise
        except (ObstoreError, OSError, TypeError, ValueError) as error:
            raise CacheBlobBackendError(
                "Obstore generation snapshot copy failed",
                context={"operation": "obstore.snapshot", "stage": "stream"},
            ) from error

    def delete_or_prove_absent(self, locator: Path | str) -> None:
        """Delete one exact generation, then settle only on exact absence proof."""
        self._require_open()
        locator_text = self._validated_locator(locator)
        try:
            self._store.delete([locator_text])
        except (NotFoundError, FileNotFoundError):
            return
        except (ObstoreError, OSError, TypeError, ValueError) as error:
            raise CacheBlobBackendError(
                "Obstore generation deletion failed",
                context={"operation": "obstore.delete", "stage": "delete"},
            ) from error
        try:
            self._store.head(locator_text)
        except (NotFoundError, FileNotFoundError):
            return
        except (ObstoreError, OSError, TypeError, ValueError) as error:
            raise CacheBlobBackendError(
                "Obstore deletion absence proof failed",
                context={"operation": "obstore.delete", "stage": "head"},
            ) from error
        raise CacheBlobBackendError(
            "Obstore generation remains present after deletion acknowledgement",
            context={"operation": "obstore.delete", "stage": "head"},
        )

    def _settle_ambiguous_publication(
        self,
        staged: GuardedStagedArtifact,
        locator: str,
        expected_size: int,
        publication_error: BaseException,
    ) -> dict[str, Any]:
        """Accept only one exact matching object after a lost create response.

        A successful exact observation never promotes anything by itself: the
        lifecycle still independently snapshots, hashes, and promotes the
        prepared generation. This narrow transport recovery only decides
        whether the immutable create effect completed.
        """

        expected_digest = self._staged_digest(staged, expected_size)
        try:
            metadata = self._store.head(locator)
        except (NotFoundError, FileNotFoundError):
            raise CacheBlobBackendError(
                "Ambiguous immutable publication remained absent",
                context={"operation": "obstore.publish", "stage": "head"},
            ) from publication_error
        except (ObstoreError, OSError, TypeError, ValueError):
            raise CacheBlobBackendError(
                "Ambiguous immutable publication could not be observed",
                context={"operation": "obstore.publish", "stage": "head"},
            ) from publication_error
        if (
            not isinstance(metadata, Mapping)
            or type(metadata.get("size")) is not int
            or metadata["size"] != expected_size
        ):
            raise CacheBlobLifecycleConflictError(
                "Ambiguous immutable publication disagrees with staged size"
            ) from publication_error

        observed_digest, observed_size = self._exact_object_identity(locator)
        if (observed_digest, observed_size) != (expected_digest, expected_size):
            raise CacheBlobLifecycleConflictError(
                "Ambiguous immutable publication disagrees with staged identity"
            ) from publication_error
        return dict(metadata)

    @staticmethod
    def _staged_digest(staged: GuardedStagedArtifact, expected_size: int) -> str:
        """Hash a second retained-descriptor view without reopening its pathname."""

        digest = hashlib.sha256()
        with staged.open() as (source, observed_size):
            if observed_size != expected_size:
                raise CacheBlobLifecycleConflictError(
                    "Staged generation size changed during publication recovery"
                )
            while chunk := source.read(64 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _exact_object_identity(self, locator: str) -> tuple[str, int]:
        """Read and hash one bounded exact object without consulting a listing."""

        try:
            result = self._store.get(locator)
        except (NotFoundError, FileNotFoundError) as error:
            raise CacheBlobBackendError(
                "Ambiguous immutable publication became absent during observation",
                context={"operation": "obstore.publish", "stage": "get"},
            ) from error
        except (ObstoreError, OSError, TypeError, ValueError) as error:
            raise CacheBlobBackendError(
                "Ambiguous immutable publication could not be read",
                context={"operation": "obstore.publish", "stage": "get"},
            ) from error

        metadata = getattr(result, "meta", None)
        if not isinstance(metadata, Mapping) or type(metadata.get("size")) is not int:
            raise CacheBlobBackendError(
                "Ambiguous immutable publication lacks a valid byte size",
                context={"operation": "obstore.publish", "stage": "get"},
            )
        declared_size = metadata["size"]
        if declared_size < 0 or declared_size > self.max_download_bytes:
            raise CacheBlobBackendError(
                "Ambiguous immutable publication exceeds configured download bound",
                context={"operation": "obstore.publish", "stage": "size"},
            )

        digest = hashlib.sha256()
        observed_size = 0
        try:
            for chunk in result.stream(min_chunk_size=1):
                if not isinstance(chunk, bytes):
                    raise CacheBlobBackendError(
                        "Obstore generation stream yielded non-bytes",
                        context={"operation": "obstore.publish", "stage": "stream"},
                    )
                observed_size += len(chunk)
                if observed_size > declared_size or observed_size > self.max_download_bytes:
                    raise CacheBlobBackendError(
                        "Ambiguous immutable publication exceeded its declared bound",
                        context={"operation": "obstore.publish", "stage": "stream"},
                    )
                digest.update(chunk)
        except CacheBlobBackendError:
            raise
        except (ObstoreError, OSError, TypeError, ValueError) as error:
            raise CacheBlobBackendError(
                "Ambiguous immutable publication stream failed",
                context={"operation": "obstore.publish", "stage": "stream"},
            ) from error
        if observed_size != declared_size:
            raise CacheBlobBackendError(
                "Ambiguous immutable publication stream disagrees with exact size",
                context={"operation": "obstore.publish", "stage": "stream"},
            )
        return digest.hexdigest(), observed_size

    def close(self) -> None:
        """Release the guarded handler resources exactly once."""
        if self._closed:
            return
        self._closed = True
        self._handler_io.close()

    def _require_open(self) -> None:
        if self._closed:
            raise CacheBlobBackendError(
                "Obstore generation participant is closed",
                context={"operation": "obstore"},
            )

    @staticmethod
    def _validated_locator(locator: Path | str) -> str:
        """Accept only canonical immutable paths below ``generations/``."""
        if not isinstance(locator, (Path, str)):
            raise CacheUnsafePathError(
                "Generation locator must be a relative path",
                reason=CacheReason.INVALID_IDENTIFIER,
            )
        text = locator.as_posix() if isinstance(locator, Path) else locator
        if (
            not text
            or text.startswith("/")
            or "\\" in text
            or ":" in text
            or "//" in text
            or text.endswith("/")
        ):
            raise CacheUnsafePathError(
                "Generation locator is not a contained canonical path",
                reason=CacheReason.INVALID_IDENTIFIER,
            )
        parts = PurePosixPath(text).parts
        if len(parts) != 3 or parts[0] != "generations" or any(
            part in {"", ".", ".."} or not _LOCATOR_SEGMENT.fullmatch(part)
            for part in parts
        ):
            raise CacheUnsafePathError(
                "Generation locator is outside the immutable namespace",
                reason=CacheReason.INVALID_IDENTIFIER,
            )
        GuardedHandlerIO.native_suffix(text)
        return text


__all__ = ["DEFAULT_MAX_TRANSFER_BYTES", "ObstoreGenerationIO"]
