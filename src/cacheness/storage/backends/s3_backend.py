"""Amazon S3 immutable-generation payload participant.

This module supplies bounded object mechanics below the shared BlobStore
lifecycle engine. Object presence is never visibility: the selected lifecycle
authority promotes verified immutable generations and owns recovery intent and
cleanup debt. The participant deliberately supports Amazon S3 through boto3
credential resolution or an injected boto3-compatible client; compatible
endpoints require separate qualification.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Iterator, Mapping

from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheConfigurationError,
    CacheReason,
    CacheUnsafePathError,
)
from cacheness.interfaces import GuardedReadSnapshot, GuardedWriteResult
from cacheness.storage.guarded_handler_io import GuardedHandlerIO, GuardedStagedArtifact
from cacheness.storage.integrity import sha256_and_size


try:  # Keep the base package importable without the optional S3 dependency.
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    BOTO3_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional installation
    boto3 = None
    BotoCoreError = Exception
    ClientError = Exception
    BOTO3_AVAILABLE = False


_MAX_LOCATOR_LENGTH = 512
_DEFAULT_CHUNK_SIZE = 1024 * 1024
_DEFAULT_DOWNLOAD_LIMIT = 128 * 1024 * 1024
_DEFAULT_DOWNLOAD_WORK_LIMIT = 4096


@dataclass(frozen=True)
class S3ObjectEvidence:
    """One bounded S3 object observation used only as reconciliation evidence."""

    locator: str
    byte_size: int


@dataclass(frozen=True)
class S3InventoryPage:
    """A single bounded continuation-token object inventory page."""

    objects: tuple[S3ObjectEvidence, ...]
    next_token: str | None


def _positive_int(name: str, value: int, *, minimum: int = 1) -> int:
    """Validate a configuration work bound without coercing surprising values."""
    if type(value) is not int or value < minimum:
        raise CacheConfigurationError(f"{name} must be an integer of at least {minimum}")
    return value


def _client_error_code(error: BaseException) -> str:
    """Extract one stable S3 service code without parsing exception text."""
    if isinstance(error, ClientError):
        response = error.response
        error_data = response.get("Error", {}) if isinstance(response, Mapping) else {}
        code = error_data.get("Code") if isinstance(error_data, Mapping) else None
        return str(code) if code is not None else ""
    return ""


def _is_absence(error: BaseException) -> bool:
    """Return whether an exact S3 operation proves only key absence."""
    return _client_error_code(error) in {"404", "NoSuchKey", "NotFound"}


class S3BlobBackend:
    """Provide bounded immutable-generation I/O for one managed Amazon S3 prefix.

    The public participant has no byte CRUD compatibility API. Callers compose
    it as a payload participant and the BlobStore engine materializes its one
    guarded I/O object through :meth:`materialize_handler_io`.
    """

    topology_capabilities = {
        "durable": True,
        "process_scope": "multi_host",
        "host_scope": "multi_host",
        "immutable_generations": True,
        "streaming": True,
        "listing": True,
    }

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "cacheness",
        region: str = "us-east-1",
        client: Any | None = None,
        expected_bucket_owner: str | None = None,
        staging_root: Path | str | None = None,
        multipart_threshold: int = 8 * 1024 * 1024,
        part_size: int = 8 * 1024 * 1024,
        max_upload_bytes: int = _DEFAULT_DOWNLOAD_LIMIT,
        max_download_bytes: int = _DEFAULT_DOWNLOAD_LIMIT,
        max_download_work: int = _DEFAULT_DOWNLOAD_WORK_LIMIT,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        max_multipart_parts: int = 10_000,
        max_upload_attempts: int = 2,
    ) -> None:
        """Configure one exact Amazon S3 bucket/prefix participant.

        Authentication uses the standard boto3 provider chain unless a caller
        injects a client. Inline access keys and endpoint URLs are intentionally
        absent: this module does not claim compatibility-service support.
        """
        if not BOTO3_AVAILABLE and client is None:
            raise ImportError(
                "boto3 is required for Amazon S3 payload storage; install cacheness[s3]"
            )
        self.bucket = self._validate_bucket(bucket)
        self.prefix = self._validate_prefix(prefix)
        if not isinstance(region, str) or not region.strip():
            raise CacheConfigurationError("region must be a non-empty string")
        self.region = region
        if expected_bucket_owner is not None and (
            not isinstance(expected_bucket_owner, str) or not expected_bucket_owner.strip()
        ):
            raise CacheConfigurationError("expected_bucket_owner must be a non-empty string")
        self.expected_bucket_owner = expected_bucket_owner
        self.multipart_threshold = _positive_int("multipart_threshold", multipart_threshold)
        self.part_size = _positive_int("part_size", part_size, minimum=5 * 1024 * 1024)
        self.max_upload_bytes = _positive_int("max_upload_bytes", max_upload_bytes)
        self.max_download_bytes = _positive_int("max_download_bytes", max_download_bytes)
        self.max_download_work = _positive_int("max_download_work", max_download_work)
        self.chunk_size = _positive_int("chunk_size", chunk_size)
        self.max_multipart_parts = _positive_int(
            "max_multipart_parts", max_multipart_parts
        )
        self.max_upload_attempts = _positive_int(
            "max_upload_attempts", max_upload_attempts
        )
        if self.max_upload_attempts > 2:
            raise CacheConfigurationError("max_upload_attempts cannot exceed two")
        if self.multipart_threshold > self.max_upload_bytes:
            raise CacheConfigurationError("multipart_threshold exceeds max_upload_bytes")

        if client is None:
            client = boto3.client("s3", region_name=region)
        self._client = client

        base = Path(staging_root) if staging_root is not None else None
        if base is not None:
            base.mkdir(parents=True, exist_ok=True)
            if not base.is_dir():
                raise CacheConfigurationError("staging_root must name a directory")
        self._temporary_root = tempfile.TemporaryDirectory(
            prefix="cacheness-s3-", dir=str(base) if base is not None else None
        )
        self._private_root = Path(self._temporary_root.name)
        self._private_root.chmod(0o700)
        self._handler_io: _S3GenerationIO | None = None
        self._closed = False

    @staticmethod
    def _validate_bucket(bucket: str) -> str:
        if (
            not isinstance(bucket, str)
            or not bucket.strip()
            or len(bucket) > 255
            or "/" in bucket
            or "\\" in bucket
            or bucket.startswith("s3:")
        ):
            raise CacheConfigurationError("bucket must be one exact S3 bucket name")
        return bucket

    @staticmethod
    def _validate_prefix(prefix: str) -> str:
        if not isinstance(prefix, str) or not prefix.strip():
            raise CacheConfigurationError("prefix must be a non-empty managed S3 prefix")
        candidate = prefix.strip().strip("/")
        if (
            not candidate
            or len(candidate) > _MAX_LOCATOR_LENGTH
            or "\\" in candidate
            or any(part in {"", ".", ".."} for part in candidate.split("/"))
        ):
            raise CacheConfigurationError("prefix must be a normalized managed S3 prefix")
        return f"{candidate}/"

    def _request_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Attach the optional expected-owner guard without adding credentials."""
        if self.expected_bucket_owner is not None:
            kwargs["ExpectedBucketOwner"] = self.expected_bucket_owner
        return kwargs

    def _locator_text(self, locator: Path | str) -> str:
        """Validate one relative generation locator below this exact prefix."""
        if not isinstance(locator, (str, Path)):
            raise CacheUnsafePathError(
                "S3 generation locator must be a relative path",
                reason=CacheReason.INVALID_IDENTIFIER,
            )
        text = str(locator)
        if (
            not text
            or len(text) > _MAX_LOCATOR_LENGTH
            or "\\" in text
            or text.startswith("s3:")
            or text.startswith("/")
            or any(part in {"", ".", ".."} for part in text.split("/"))
        ):
            raise CacheUnsafePathError(
                "S3 generation locator is outside the managed prefix",
                reason=CacheReason.INVALID_IDENTIFIER,
            )
        normalized = PurePosixPath(text)
        if normalized.is_absolute() or normalized.parts != tuple(text.split("/")):
            raise CacheUnsafePathError(
                "S3 generation locator is not normalized",
                reason=CacheReason.INVALID_IDENTIFIER,
            )
        return normalized.as_posix()

    def _object_key(self, locator: Path | str) -> str:
        """Map one validated relative generation locator beneath the managed prefix."""
        return f"{self.prefix}{self._locator_text(locator)}"

    def materialize_handler_io(self) -> "_S3GenerationIO":
        """Return the one guarded primitive consumed by the lifecycle engine."""
        if self._closed:
            raise RuntimeError("S3 payload participant is closed")
        if self._handler_io is None:
            self._handler_io = _S3GenerationIO(self)
        return self._handler_io

    def close(self) -> None:
        """Release private local staging resources; boto3 owns its connection pool."""
        if not self._closed:
            if self._handler_io is not None:
                self._handler_io.close()
            self._temporary_root.cleanup()
            self._closed = True


class _S3GenerationIO:
    """Guarded handler I/O backed by immutable objects below one S3 prefix."""

    def __init__(self, backend: S3BlobBackend) -> None:
        self._backend = backend
        self._staging = GuardedHandlerIO(backend._private_root)
        self._closed = False

    def _require_open(self) -> None:
        if self._closed or self._backend._closed:
            raise RuntimeError("S3 generation I/O is closed")

    @contextmanager
    def stage(
        self, handler: Any, data: Any, config: Any
    ) -> Iterator[GuardedStagedArtifact]:
        """Stage one handler-native file privately before authority intent exists."""
        self._require_open()
        with self._staging.stage(handler, data, config) as staged:
            yield staged

    @staticmethod
    def _stage_digest(source: Any, expected_size: int) -> str:
        """Hash one validated staged file and restore it for a single request upload."""
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = source.read(_DEFAULT_CHUNK_SIZE)
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise CacheBlobBackendError("S3 staged payload stream returned non-bytes")
            observed += len(chunk)
            if observed > expected_size:
                raise CacheBlobBackendError("S3 staged payload size changed before upload")
            digest.update(chunk)
        if observed != expected_size:
            raise CacheBlobBackendError("S3 staged payload size changed before upload")
        source.seek(0)
        return digest.hexdigest()

    def publish_generation(
        self, staged: GuardedStagedArtifact, locator: Path | str
    ) -> GuardedWriteResult:
        """Conditionally create one immutable S3 generation under explicit bounds."""
        self._require_open()
        locator_text = self._backend._locator_text(locator)
        key = self._backend._object_key(locator_text)
        with staged.open() as (source, file_size):
            if file_size > self._backend.max_upload_bytes:
                raise CacheBlobBackendError(
                    "S3 staged payload exceeds configured upload bound",
                    context={"operation": "s3.publish", "stage": "validate"},
                )
            digest = self._stage_digest(source, file_size)
            if file_size > self._backend.multipart_threshold:
                self._publish_multipart(source, key, locator_text, file_size, digest)
            else:
                self._publish_single(source, key, locator_text, file_size, digest)
        return staged.result_for(Path(locator_text), file_size)

    def _publish_single(
        self,
        source: Any,
        key: str,
        locator: str,
        file_size: int,
        digest: str,
    ) -> None:
        """Create one small generation with an S3 conditional write."""
        try:
            self._backend._client.put_object(
                **self._backend._request_kwargs(
                    Bucket=self._backend.bucket,
                    Key=key,
                    Body=source,
                    ContentLength=file_size,
                    IfNoneMatch="*",
                )
            )
        except ClientError as error:
            if _client_error_code(error) in {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"}:
                self._resolve_ambiguous_publication(locator, digest, file_size, error)
                return
            raise CacheBlobBackendError(
                "S3 conditional generation publication failed",
                context={"operation": "s3.publish", "stage": "conditional_put"},
            ) from error
        except BotoCoreError as error:
            self._resolve_ambiguous_publication(locator, digest, file_size, error)

    def _publish_multipart(
        self,
        source: Any,
        key: str,
        locator: str,
        file_size: int,
        digest: str,
    ) -> None:
        """Publish bounded parts and conditionally complete one immutable generation."""
        part_count = (file_size + self._backend.part_size - 1) // self._backend.part_size
        if part_count > self._backend.max_multipart_parts:
            raise CacheBlobBackendError(
                "S3 staged payload exceeds configured multipart part bound",
                context={"operation": "s3.publish", "stage": "validate"},
            )
        for attempt in range(self._backend.max_upload_attempts):
            upload_id = self._create_multipart_upload(key)
            try:
                parts = self._upload_parts(source, key, upload_id, file_size, part_count)
            except (ClientError, BotoCoreError, OSError, TypeError, ValueError) as error:
                self._abort_known_upload(key, upload_id, error)
                raise CacheBlobBackendError(
                    "S3 multipart generation publication failed before completion",
                    context={"operation": "s3.publish", "stage": "upload_part"},
                ) from error
            try:
                self._backend._client.complete_multipart_upload(
                    **self._backend._request_kwargs(
                        Bucket=self._backend.bucket,
                        Key=key,
                        UploadId=upload_id,
                        MultipartUpload={"Parts": parts},
                        IfNoneMatch="*",
                    )
                )
                return
            except ClientError as error:
                code = _client_error_code(error)
                if code in {"409", "ConditionalRequestConflict"}:
                    self._abort_known_upload(key, upload_id, error)
                    if attempt + 1 < self._backend.max_upload_attempts:
                        continue
                    raise CacheBlobBackendError(
                        "S3 multipart conditional completion remained conflicted",
                        context={"operation": "s3.publish", "stage": "complete"},
                    ) from error
                if code in {"412", "PreconditionFailed"}:
                    self._abort_known_upload(key, upload_id, error)
                self._resolve_ambiguous_publication(locator, digest, file_size, error)
                return
            except BotoCoreError as error:
                self._resolve_ambiguous_publication(locator, digest, file_size, error)
                return
        raise CacheBlobBackendError(
            "S3 multipart upload exhausted its bounded attempts",
            context={"operation": "s3.publish", "stage": "complete"},
        )

    def _create_multipart_upload(self, key: str) -> str:
        """Create one exact, bounded multipart session without lifecycle state."""
        try:
            response = self._backend._client.create_multipart_upload(
                **self._backend._request_kwargs(Bucket=self._backend.bucket, Key=key)
            )
        except (ClientError, BotoCoreError) as error:
            raise CacheBlobBackendError(
                "S3 multipart creation failed",
                context={"operation": "s3.publish", "stage": "create_multipart"},
            ) from error
        upload_id = response.get("UploadId") if isinstance(response, Mapping) else None
        if not isinstance(upload_id, str) or not upload_id:
            raise CacheBlobBackendError(
                "S3 multipart creation returned no upload identity",
                context={"operation": "s3.publish", "stage": "create_multipart"},
            )
        return upload_id

    def _upload_parts(
        self,
        source: Any,
        key: str,
        upload_id: str,
        file_size: int,
        part_count: int,
    ) -> list[dict[str, Any]]:
        """Upload exactly the validated staged bytes in bounded parts."""
        source.seek(0)
        remaining = file_size
        parts: list[dict[str, Any]] = []
        for number in range(1, part_count + 1):
            chunk = source.read(min(self._backend.part_size, remaining))
            if not isinstance(chunk, bytes) or not chunk:
                raise OSError("S3 staged payload ended before multipart upload completed")
            remaining -= len(chunk)
            response = self._backend._client.upload_part(
                **self._backend._request_kwargs(
                    Bucket=self._backend.bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=number,
                    Body=chunk,
                )
            )
            etag = response.get("ETag") if isinstance(response, Mapping) else None
            if not isinstance(etag, str) or not etag:
                raise OSError("S3 multipart part response lacks its protocol ETag")
            parts.append({"PartNumber": number, "ETag": etag})
        if remaining != 0 or source.read(1):
            raise OSError("S3 staged payload changed during multipart upload")
        return parts

    def _abort_known_upload(
        self, key: str, upload_id: str, prior_error: BaseException
    ) -> None:
        """Abort one known pre-completion upload; never list to discover a target."""
        try:
            self._backend._client.abort_multipart_upload(
                **self._backend._request_kwargs(
                    Bucket=self._backend.bucket,
                    Key=key,
                    UploadId=upload_id,
                )
            )
        except (ClientError, BotoCoreError) as cleanup_error:
            raise CacheBlobBackendError(
                "S3 multipart failure left an upload requiring explicit reconciliation",
                context={"operation": "s3.publish", "stage": "abort_multipart"},
            ) from cleanup_error

    def _resolve_ambiguous_publication(
        self,
        locator: str,
        digest: str,
        byte_size: int,
        cause: BaseException,
    ) -> None:
        """Classify a lost/conditional response from the exact key and manifest proof."""
        try:
            with self.open_snapshot(locator, {}) as snapshot:
                actual_digest, actual_size = sha256_and_size(snapshot.path)
        except FileNotFoundError:
            raise CacheBlobBackendError(
                "S3 publication response was ambiguous and exact key is absent",
                context={"operation": "s3.publish", "stage": "classify"},
            ) from cause
        if actual_digest == digest and actual_size == byte_size:
            return
        raise CacheBlobLifecycleConflictError(
            "S3 immutable generation conflicts with staged manifest bytes",
            context={"operation": "s3.publish", "stage": "classify"},
        ) from cause

    @contextmanager
    def open_snapshot(
        self, locator: Path | str, metadata: Mapping[str, Any]
    ) -> Iterator[GuardedReadSnapshot]:
        """Stream one exact object into a bounded private snapshot and close it."""
        self._require_open()
        locator_text = self._backend._locator_text(locator)
        key = self._backend._object_key(locator_text)
        try:
            head = self._backend._client.head_object(
                **self._backend._request_kwargs(Bucket=self._backend.bucket, Key=key)
            )
        except ClientError as error:
            if _is_absence(error):
                raise FileNotFoundError("S3 generation is absent") from error
            raise CacheBlobBackendError(
                "S3 generation preflight failed",
                context={"operation": "s3.snapshot", "stage": "head"},
            ) from error
        except BotoCoreError as error:
            raise CacheBlobBackendError(
                "S3 generation preflight was not confirmed",
                context={"operation": "s3.snapshot", "stage": "head"},
            ) from error
        if not isinstance(head, Mapping) or type(head.get("ContentLength")) is not int:
            raise CacheBlobBackendError(
                "S3 generation preflight returned an invalid content length",
                context={"operation": "s3.snapshot", "stage": "head"},
            )
        expected_size = head["ContentLength"]
        if expected_size < 0 or expected_size > self._backend.max_download_bytes:
            raise CacheBlobBackendError(
                "S3 generation exceeds configured snapshot bound",
                context={"operation": "s3.snapshot", "stage": "head"},
            )
        try:
            response = self._backend._client.get_object(
                **self._backend._request_kwargs(Bucket=self._backend.bucket, Key=key)
            )
        except ClientError as error:
            if _is_absence(error):
                raise FileNotFoundError("S3 generation is absent") from error
            raise CacheBlobBackendError(
                "S3 generation read request failed",
                context={"operation": "s3.snapshot", "stage": "get"},
            ) from error
        except BotoCoreError as error:
            raise CacheBlobBackendError(
                "S3 generation read request was not confirmed",
                context={"operation": "s3.snapshot", "stage": "get"},
            ) from error
        if not isinstance(response, Mapping) or not callable(getattr(response.get("Body"), "read", None)):
            raise CacheBlobBackendError(
                "S3 generation response lacks a streaming body",
                context={"operation": "s3.snapshot", "stage": "get"},
            )
        body = response["Body"]
        suffix = "".join(Path(locator_text).suffixes)
        if len(suffix) > 96:
            raise CacheUnsafePathError(
                "S3 generation suffix exceeds the private snapshot bound",
                reason=CacheReason.INVALID_IDENTIFIER,
            )
        with tempfile.TemporaryDirectory(
            prefix="cacheness-s3-read-", dir=self._backend._private_root
        ) as temporary:
            snapshot_path = Path(temporary) / f"snapshot{suffix}"
            observed = 0
            work = 0
            file_descriptor: int | None = None
            destination = None
            try:
                file_descriptor = os.open(
                    snapshot_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                destination = os.fdopen(file_descriptor, "wb")
                file_descriptor = None
                while observed < expected_size:
                    work += 1
                    if work > self._backend.max_download_work:
                        raise CacheBlobBackendError(
                            "S3 generation exceeded configured snapshot work bound",
                            context={"operation": "s3.snapshot", "stage": "stream"},
                        )
                    chunk = body.read(min(self._backend.chunk_size, expected_size - observed + 1))
                    if not isinstance(chunk, bytes) or not chunk:
                        raise CacheBlobBackendError(
                            "S3 generation stream ended before declared length",
                            context={"operation": "s3.snapshot", "stage": "stream"},
                        )
                    observed += len(chunk)
                    if observed > expected_size:
                        raise CacheBlobBackendError(
                            "S3 generation stream exceeded declared length",
                            context={"operation": "s3.snapshot", "stage": "stream"},
                        )
                    destination.write(chunk)
                work += 1
                if work > self._backend.max_download_work:
                    raise CacheBlobBackendError(
                        "S3 generation exceeded configured snapshot work bound",
                        context={"operation": "s3.snapshot", "stage": "stream"},
                    )
                if body.read(1):
                    raise CacheBlobBackendError(
                        "S3 generation stream exceeded declared length",
                        context={"operation": "s3.snapshot", "stage": "stream"},
                    )
                destination.flush()
                os.fsync(destination.fileno())
            except CacheBlobBackendError:
                raise
            except (OSError, TypeError, ValueError) as error:
                raise CacheBlobBackendError(
                    "S3 generation snapshot copy failed",
                    context={"operation": "s3.snapshot", "stage": "stream"},
                ) from error
            finally:
                if destination is not None:
                    destination.close()
                elif file_descriptor is not None:
                    os.close(file_descriptor)
                close = getattr(body, "close", None)
                if callable(close):
                    close()
            snapshot_metadata = dict(metadata)
            snapshot_metadata["actual_path"] = str(snapshot_path)
            yield GuardedReadSnapshot(snapshot_path, snapshot_metadata)

    def delete_or_prove_absent(self, locator: Path | str) -> None:
        """Delete one exact generation; final absence proof is added with inventory work."""
        self._require_open()
        key = self._backend._object_key(locator)
        try:
            self._backend._client.delete_object(
                **self._backend._request_kwargs(Bucket=self._backend.bucket, Key=key)
            )
        except (ClientError, BotoCoreError) as error:
            raise CacheBlobBackendError(
                "S3 generation deletion was not confirmed",
                context={"operation": "s3.delete", "stage": "delete"},
            ) from error

    def close(self) -> None:
        """Release only local guarded staging resources."""
        if not self._closed:
            self._staging.close()
            self._closed = True


__all__ = [
    "BOTO3_AVAILABLE",
    "S3BlobBackend",
    "S3InventoryPage",
    "S3ObjectEvidence",
]
