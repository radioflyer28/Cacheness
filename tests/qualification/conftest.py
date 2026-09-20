"""Externally configured fixtures for the real Phase 5 qualification suite.

This module never discovers, creates, or deletes cloud/database accounts.  A
qualification run owns one random schema and one random prefix, marked before
any cleanup is permitted.  The runner calls the cleanup function again as a
backstop if pytest is interrupted.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
import re
import secrets
from typing import Any, Mapping
from urllib.parse import urlparse

import pytest


_RUN_ID = re.compile(r"phase[58]-[0-9a-f]{32}\Z")
_OWNER_TABLE = "__cacheness_qualification_owner"
_OWNER_MARKER_NAME = "__cacheness_qualification_owner__.json"
_MAX_OWNER_MARKER_BYTES = 512
_MAX_CLEANUP_PAGES = 10
_MAX_CLEANUP_OBJECTS = 1_000
_MAX_CLEANUP_BYTES = 64 * 1024 * 1024
_MAX_DELETE_BATCH = 100
_MAX_MULTIPART_UPLOADS = 1_000
_CONNECT_TIMEOUT_SECONDS = 5


class QualificationConfigurationError(ValueError):
    """External qualification configuration is absent or invalid."""


@dataclass(frozen=True)
class QualificationConfig:
    """Secret-bearing values held in memory only for one live test process."""

    postgres_dsn: str
    s3_bucket: str
    manifest_key: bytes
    region: str


@dataclass(frozen=True)
class QualificationNamespace:
    """The exact service resources that one runner invocation is allowed to own."""

    run_id: str
    schema: str
    prefix: str

    @property
    def s3_owner_marker_key(self) -> str:
        """Return the one object that authorizes exact-prefix cleanup."""
        return f"{self.prefix}{_OWNER_MARKER_NAME}"


class InMemoryManifestSigner:
    """A narrow application-owned signer provider with no filesystem persistence."""

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, bytes) or len(key) != 32:
            raise QualificationConfigurationError("invalid manifest signing material")
        self._key = key

    def get_key(self) -> bytes:
        """Supply the already-decoded caller key to one BlobStore client."""
        return self._key

    def __repr__(self) -> str:
        return "InMemoryManifestSigner(redacted=True)"


@dataclass
class LiveQualificationResources:
    """Live test resources, all bounded by the immutable run namespace."""

    config: QualificationConfig
    namespace: QualificationNamespace
    authority: Any
    s3_client: Any
    first_signer: InMemoryManifestSigner
    second_signer: InMemoryManifestSigner


def qualification_config_from_environment(
    environment: Mapping[str, str],
) -> QualificationConfig:
    """Decode external configuration without writing or echoing secret values."""
    required = (
        "CACHENESS_TEST_POSTGRES_DSN",
        "CACHENESS_TEST_S3_BUCKET",
        "CACHENESS_TEST_MANIFEST_KEY_B64",
        "CACHENESS_TEST_AWS_REGION",
    )
    missing = tuple(name for name in required if not environment.get(name))
    if "CACHENESS_TEST_AWS_REGION" in missing:
        raise QualificationConfigurationError("explicit AWS region is required")
    if missing:
        raise QualificationConfigurationError("required external configuration is absent")
    try:
        key = base64.b64decode(
            environment["CACHENESS_TEST_MANIFEST_KEY_B64"], validate=True
        )
    except Exception as error:
        raise QualificationConfigurationError("invalid manifest signing material") from error
    if len(key) != 32:
        raise QualificationConfigurationError("invalid manifest signing material")
    return QualificationConfig(
        postgres_dsn=environment["CACHENESS_TEST_POSTGRES_DSN"],
        s3_bucket=environment["CACHENESS_TEST_S3_BUCKET"],
        manifest_key=key,
        region=environment["CACHENESS_TEST_AWS_REGION"],
    )


def qualification_namespace(run_id: str | None = None) -> QualificationNamespace:
    """Derive bounded PostgreSQL/S3 names from one cryptographically random ID."""
    selected_run_id = run_id or f"phase5-{secrets.token_hex(16)}"
    if not _RUN_ID.fullmatch(selected_run_id):
        raise QualificationConfigurationError("invalid qualification run identifier")
    phase = selected_run_id[5]
    token = selected_run_id.removeprefix(f"phase{phase}-")
    return QualificationNamespace(
        run_id=selected_run_id,
        schema=f"cacheness_q{phase}_{token}",
        prefix=f"cacheness-qualification/{selected_run_id}/",
    )


def _qualification_run_id_from_environment(environment: Mapping[str, str]) -> str | None:
    """Select one explicit runner namespace without mixing qualification generations."""
    phase5_run_id = environment.get("CACHENESS_PHASE5_QUALIFICATION_RUN_ID")
    phase8_run_id = environment.get("CACHENESS_PHASE8_QUALIFICATION_RUN_ID")
    if phase5_run_id and phase8_run_id:
        raise QualificationConfigurationError("multiple qualification run identifiers")
    return phase8_run_id or phase5_run_id


def independent_manifest_signers(
    config: QualificationConfig,
) -> tuple[InMemoryManifestSigner, InMemoryManifestSigner]:
    """Build distinct providers from the same externally supplied shared bytes."""
    return InMemoryManifestSigner(config.manifest_key), InMemoryManifestSigner(
        config.manifest_key
    )


def _owner_marker_payload(namespace: QualificationNamespace) -> bytes:
    """Return only the exact run identifier required for cleanup authorization."""
    return (json.dumps({"run_id": namespace.run_id}, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _s3_owner_request(bucket: str) -> dict[str, str]:
    """Bind fixture administration to the configured exact bucket only."""
    return {"Bucket": bucket}


def _s3_marker_matches(
    client: Any,
    bucket: str,
    namespace: QualificationNamespace,
) -> bool:
    """Read at most one small exact-key marker before any destructive operation."""
    try:
        response = client.get_object(
            **_s3_owner_request(bucket),
            Key=namespace.s3_owner_marker_key,
        )
        body = response["Body"]
        raw = body.read(_MAX_OWNER_MARKER_BYTES + 1)
        body.close()
    except Exception:
        return False
    if not isinstance(raw, bytes) or len(raw) > _MAX_OWNER_MARKER_BYTES:
        return False
    return raw == _owner_marker_payload(namespace)


def _s3_prefix_has_residue(
    client: Any,
    bucket: str,
    namespace: QualificationNamespace,
) -> bool:
    """Check only the exact prefix, treating failures as unresolved residue."""
    try:
        objects = client.list_objects_v2(
            **_s3_owner_request(bucket),
            Prefix=namespace.prefix,
            MaxKeys=1,
        ).get("Contents", [])
        uploads = client.list_multipart_uploads(
            **_s3_owner_request(bucket),
            Prefix=namespace.prefix,
            MaxUploads=1,
        ).get("Uploads", [])
    except Exception:
        return True
    return bool(objects or uploads)


def _bounded_delete_prefix(
    client: Any,
    bucket: str,
    namespace: QualificationNamespace,
) -> bool:
    """Delete only a marker-authorized prefix under fixed page/object/byte bounds."""
    continuation: str | None = None
    pages = 0
    objects_seen = 0
    bytes_seen = 0
    while True:
        if pages >= _MAX_CLEANUP_PAGES:
            return False
        request: dict[str, object] = {
            **_s3_owner_request(bucket),
            "Prefix": namespace.prefix,
            "MaxKeys": _MAX_DELETE_BATCH,
        }
        if continuation is not None:
            request["ContinuationToken"] = continuation
        try:
            response = client.list_objects_v2(**request)
        except Exception:
            return False
        pages += 1
        contents = response.get("Contents", [])
        if not isinstance(contents, list):
            return False
        objects: list[dict[str, str]] = []
        for item in contents:
            if not isinstance(item, Mapping):
                return False
            key = item.get("Key")
            size = item.get("Size")
            if not isinstance(key, str) or not key.startswith(namespace.prefix):
                return False
            if type(size) is not int or size < 0:
                return False
            # The marker is the only authorization for this exact run.  Keep
            # it out of bulk deletion so any interrupted later page remains
            # safely retryable through this same guarded cleanup path.
            if key == namespace.s3_owner_marker_key:
                continue
            objects_seen += 1
            bytes_seen += size
            if (
                objects_seen > _MAX_CLEANUP_OBJECTS
                or bytes_seen > _MAX_CLEANUP_BYTES
            ):
                return False
            objects.append({"Key": key})
        if objects:
            try:
                deleted = client.delete_objects(
                    **_s3_owner_request(bucket),
                    Delete={"Objects": objects, "Quiet": True},
                )
            except Exception:
                return False
            if deleted.get("Errors"):
                return False
        truncated = response.get("IsTruncated", False)
        if truncated is not True:
            return True
        next_token = response.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token or len(next_token) > 1_024:
            return False
        continuation = next_token


def _s3_prefix_contains_only_owner_marker(
    client: Any,
    bucket: str,
    namespace: QualificationNamespace,
) -> bool:
    """Prove bounded S3 cleanup reached only its exact authorization marker."""
    continuation: str | None = None
    pages = 0
    objects_seen = 0
    bytes_seen = 0
    marker_seen = False
    while True:
        if pages >= _MAX_CLEANUP_PAGES:
            return False
        request: dict[str, object] = {
            **_s3_owner_request(bucket),
            "Prefix": namespace.prefix,
            "MaxKeys": _MAX_DELETE_BATCH,
        }
        if continuation is not None:
            request["ContinuationToken"] = continuation
        try:
            response = client.list_objects_v2(**request)
        except Exception:
            return False
        pages += 1
        contents = response.get("Contents", [])
        if not isinstance(contents, list):
            return False
        for item in contents:
            if not isinstance(item, Mapping):
                return False
            key = item.get("Key")
            size = item.get("Size")
            if not isinstance(key, str) or not key.startswith(namespace.prefix):
                return False
            if type(size) is not int or size < 0:
                return False
            objects_seen += 1
            bytes_seen += size
            if (
                objects_seen > _MAX_CLEANUP_OBJECTS
                or bytes_seen > _MAX_CLEANUP_BYTES
                or key != namespace.s3_owner_marker_key
            ):
                return False
            marker_seen = True
        if response.get("IsTruncated", False) is not True:
            break
        next_token = response.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token or len(next_token) > 1_024:
            return False
        continuation = next_token

    key_marker: str | None = None
    upload_id_marker: str | None = None
    upload_pages = 0
    while True:
        if upload_pages >= _MAX_CLEANUP_PAGES:
            return False
        request = {
            **_s3_owner_request(bucket),
            "Prefix": namespace.prefix,
            "MaxUploads": _MAX_DELETE_BATCH,
        }
        if key_marker is not None:
            request["KeyMarker"] = key_marker
        if upload_id_marker is not None:
            request["UploadIdMarker"] = upload_id_marker
        try:
            response = client.list_multipart_uploads(**request)
        except Exception:
            return False
        upload_pages += 1
        uploads = response.get("Uploads", [])
        if not isinstance(uploads, list) or uploads:
            return False
        if response.get("IsTruncated", False) is not True:
            return marker_seen
        next_key = response.get("NextKeyMarker")
        next_upload = response.get("NextUploadIdMarker")
        if not isinstance(next_key, str) or not isinstance(next_upload, str):
            return False
        key_marker, upload_id_marker = next_key, next_upload


def _delete_s3_owner_marker(
    client: Any,
    bucket: str,
    namespace: QualificationNamespace,
) -> bool:
    """Delete the exact marker only after bounded cleanup proved it is alone."""
    try:
        client.delete_object(
            **_s3_owner_request(bucket),
            Key=namespace.s3_owner_marker_key,
        )
    except Exception:
        return False
    return True


def _bounded_abort_multipart_uploads(
    client: Any,
    bucket: str,
    namespace: QualificationNamespace,
) -> bool:
    """Abort only observed uploads under the marker-authorized exact prefix."""
    key_marker: str | None = None
    upload_id_marker: str | None = None
    pages = 0
    uploads_seen = 0
    while True:
        if pages >= _MAX_CLEANUP_PAGES:
            return False
        request: dict[str, object] = {
            **_s3_owner_request(bucket),
            "Prefix": namespace.prefix,
            "MaxUploads": _MAX_DELETE_BATCH,
        }
        if key_marker is not None:
            request["KeyMarker"] = key_marker
        if upload_id_marker is not None:
            request["UploadIdMarker"] = upload_id_marker
        try:
            response = client.list_multipart_uploads(**request)
        except Exception:
            return False
        pages += 1
        uploads = response.get("Uploads", [])
        if not isinstance(uploads, list):
            return False
        for upload in uploads:
            if not isinstance(upload, Mapping):
                return False
            key = upload.get("Key")
            upload_id = upload.get("UploadId")
            if (
                not isinstance(key, str)
                or not key.startswith(namespace.prefix)
                or not isinstance(upload_id, str)
                or not upload_id
            ):
                return False
            uploads_seen += 1
            if uploads_seen > _MAX_MULTIPART_UPLOADS:
                return False
            try:
                client.abort_multipart_upload(
                    **_s3_owner_request(bucket),
                    Key=key,
                    UploadId=upload_id,
                )
            except Exception:
                return False
        if response.get("IsTruncated", False) is not True:
            return True
        next_key = response.get("NextKeyMarker")
        next_upload = response.get("NextUploadIdMarker")
        if not isinstance(next_key, str) or not isinstance(next_upload, str):
            return False
        key_marker, upload_id_marker = next_key, next_upload


def cleanup_s3_run(
    client: Any,
    bucket: str,
    namespace: QualificationNamespace,
) -> str:
    """Clean one exact S3 run or return residue without crossing its prefix."""
    marker_matches = _s3_marker_matches(client, bucket, namespace)
    if not marker_matches:
        return (
            "CLEAN"
            if not _s3_prefix_has_residue(client, bucket, namespace)
            else "RESIDUE"
        )
    if not _bounded_abort_multipart_uploads(client, bucket, namespace):
        return "RESIDUE"
    if not _bounded_delete_prefix(client, bucket, namespace):
        return "RESIDUE"
    if not _s3_prefix_contains_only_owner_marker(client, bucket, namespace):
        return "RESIDUE"
    if not _delete_s3_owner_marker(client, bucket, namespace):
        return "RESIDUE"
    return (
        "CLEAN"
        if not _s3_prefix_has_residue(client, bucket, namespace)
        else "RESIDUE"
    )


def _postgresql_connection(config: QualificationConfig) -> Any:
    """Open one short-lived externally configured PostgreSQL connection."""
    try:
        import psycopg
    except ImportError as error:
        raise QualificationConfigurationError("PostgreSQL driver is unavailable") from error
    return psycopg.connect(config.postgres_dsn, connect_timeout=_CONNECT_TIMEOUT_SECONDS)


def _postgresql_sql() -> Any:
    """Load identifier composition only when the live fixture needs it."""
    try:
        from psycopg import sql
    except ImportError as error:
        raise QualificationConfigurationError("PostgreSQL driver is unavailable") from error
    return sql


def _postgresql_schema_exists(cursor: Any, namespace: QualificationNamespace) -> bool:
    cursor.execute(
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
        (namespace.schema,),
    )
    return cursor.fetchone() is not None


def _postgresql_owner_matches(cursor: Any, namespace: QualificationNamespace) -> bool:
    sql = _postgresql_sql()
    cursor.execute(
        sql.SQL("SELECT run_id FROM {}.{} LIMIT 2").format(
            sql.Identifier(namespace.schema), sql.Identifier(_OWNER_TABLE)
        )
    )
    rows = cursor.fetchall()
    return len(rows) == 1 and rows[0] == (namespace.run_id,)


def cleanup_postgresql_run(config: QualificationConfig, namespace: QualificationNamespace) -> str:
    """Drop only a schema whose exact ownership marker still matches this run."""
    connection: Any | None = None
    try:
        connection = _postgresql_connection(config)
        with connection.cursor() as cursor:
            if not _postgresql_schema_exists(cursor, namespace):
                return "CLEAN"
            if not _postgresql_owner_matches(cursor, namespace):
                return "RESIDUE"
            sql = _postgresql_sql()
            cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(namespace.schema)))
        connection.commit()
        return "CLEAN"
    except Exception:
        if connection is not None:
            try:
                connection.rollback()
            except Exception:
                pass
        return "RESIDUE"
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def _s3_client(config: QualificationConfig) -> Any:
    """Construct the standard-provider Amazon S3 client without endpoint overrides."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as error:
        raise QualificationConfigurationError("AWS SDK is unavailable") from error
    session = boto3.Session(region_name=config.region)
    client = session.client(
        "s3",
        config=Config(
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
            read_timeout=_CONNECT_TIMEOUT_SECONDS,
            retries={"max_attempts": 1, "mode": "standard"},
        ),
    )
    hostname = urlparse(client.meta.endpoint_url).hostname
    if hostname is None or not (
        hostname == "s3.amazonaws.com" or hostname.endswith(".amazonaws.com")
    ):
        raise QualificationConfigurationError("non-Amazon S3 endpoint is not eligible")
    return client


def _create_postgresql_namespace(config: QualificationConfig, namespace: QualificationNamespace) -> None:
    """Create and mark one schema before the authority initializes its tables."""
    connection = _postgresql_connection(config)
    try:
        sql = _postgresql_sql()
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace.schema)))
            cursor.execute(
                sql.SQL("CREATE TABLE {}.{} (run_id TEXT PRIMARY KEY NOT NULL)").format(
                    sql.Identifier(namespace.schema), sql.Identifier(_OWNER_TABLE)
                )
            )
            cursor.execute(
                sql.SQL("INSERT INTO {}.{} (run_id) VALUES (%s)").format(
                    sql.Identifier(namespace.schema), sql.Identifier(_OWNER_TABLE)
                ),
                (namespace.run_id,),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _create_s3_owner_marker(client: Any, config: QualificationConfig, namespace: QualificationNamespace) -> None:
    """Write the only cleanup authorization object inside the owned prefix."""
    request: dict[str, object] = {
        "Bucket": config.s3_bucket,
        "Key": namespace.s3_owner_marker_key,
        "Body": _owner_marker_payload(namespace),
        "ContentType": "application/json",
    }
    client.put_object(**request)


def create_live_qualification_resources(
    environment: Mapping[str, str],
) -> LiveQualificationResources:
    """Create only the marked schema/prefix and explicitly initialize authority state."""
    config = qualification_config_from_environment(environment)
    namespace = qualification_namespace(_qualification_run_id_from_environment(environment))
    first_signer, second_signer = independent_manifest_signers(config)
    _create_postgresql_namespace(config, namespace)
    authority: Any | None = None
    s3_client: Any | None = None
    try:
        from cacheness.storage.backends.postgresql_lifecycle_authority import (
            PostgresqlLifecycleAuthority,
        )

        authority = PostgresqlLifecycleAuthority(
            lambda: _postgresql_connection(config),
            schema=namespace.schema,
            store_identity=f"phase5-{namespace.run_id[-32:]}",
        )
        authority.initialize()
        s3_client = _s3_client(config)
        _create_s3_owner_marker(s3_client, config, namespace)
        return LiveQualificationResources(
            config=config,
            namespace=namespace,
            authority=authority,
            s3_client=s3_client,
            first_signer=first_signer,
            second_signer=second_signer,
        )
    except Exception:
        if authority is not None:
            authority.close()
        cleanup_qualification_resources(environment, namespace.run_id)
        raise


def cleanup_qualification_resources(environment: Mapping[str, str], run_id: str) -> str:
    """Attempt bounded cleanup for exactly one runner-created namespace."""
    try:
        config = qualification_config_from_environment(environment)
        namespace = qualification_namespace(run_id)
        client = _s3_client(config)
    except Exception:
        return "RESIDUE"
    postgres_status = cleanup_postgresql_run(config, namespace)
    s3_status = cleanup_s3_run(client, config.s3_bucket, namespace)
    return "CLEAN" if postgres_status == s3_status == "CLEAN" else "RESIDUE"


@pytest.fixture(scope="session")
def live_qualification_resources() -> LiveQualificationResources:
    """Provide one externally configured, runner-owned real-service fixture set."""
    environment = dict(os.environ)
    try:
        resources = create_live_qualification_resources(environment)
    except QualificationConfigurationError as error:
        pytest.fail(str(error), pytrace=False)
    except Exception:
        pytest.fail("live qualification resource setup failed", pytrace=False)
    try:
        yield resources
    finally:
        try:
            resources.authority.close()
        finally:
            status = cleanup_qualification_resources(environment, resources.namespace.run_id)
        if status != "CLEAN":
            pytest.fail("live qualification cleanup left residue", pytrace=False)


@pytest.fixture(scope="session")
def live_postgresql_authority(live_qualification_resources: LiveQualificationResources) -> Any:
    """Expose the initialized authority without exposing its DSN."""
    return live_qualification_resources.authority


@pytest.fixture(scope="session")
def live_s3_client(live_qualification_resources: LiveQualificationResources) -> Any:
    """Expose one standard-provider S3 client bound to the owned prefix fixture."""
    return live_qualification_resources.s3_client


@pytest.fixture(scope="session")
def live_manifest_signers(
    live_qualification_resources: LiveQualificationResources,
) -> tuple[InMemoryManifestSigner, InMemoryManifestSigner]:
    """Provide distinct signer providers for independent remote clients."""
    return (
        live_qualification_resources.first_signer,
        live_qualification_resources.second_signer,
    )
