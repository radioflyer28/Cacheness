"""Contract tests for canonical schema-1 BlobStore manifest records."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from cacheness.config import CacheConfig, CompressionConfig
from cacheness.error_handling import (
    CacheManifestIntegrityError,
    CacheManifestUnsupportedVersionError,
)
from cacheness.handlers import (
    BLOSC2_AVAILABLE,
    ArrayHandler,
    HandlerRegistry,
    ObjectHandler,
    PANDAS_AVAILABLE,
    PandasDataFrameHandler,
)
from cacheness.storage import BlobManifestV1
from cacheness.storage.integrity import sign_hmac_sha256, verify_hmac_sha256
from cacheness.storage.manifest import (
    MAX_COLLECTION_ITEMS,
    MAX_MANIFEST_BYTES,
    MAX_NESTING_DEPTH,
    MAX_STRING_UTF8_BYTES,
    MAX_TOTAL_NODES,
)


_KEY = b"canonical-manifest-test-key-0001"


def _manifest(**overrides) -> BlobManifestV1:
    """Build one complete signed schema-1 manifest for codec contracts."""
    values = {
        "schema_version": 1,
        "key": "tenant/asset",
        "generation": "0123456789abcdef",
        "state": "committed",
        "locator": "/managed/payload.bin",
        "handler_type": "object",
        "payload_format": "compressed_pickle",
        "payload_format_version": 1,
        "digest_algorithm": "sha256",
        "digest": "a" * 64,
        "byte_size": 7,
        "created_at": "2026-08-30T00:00:00+00:00",
        "handler_metadata": {"serializer": "pickle", "empty": ""},
        "user_metadata": {"label": "café", "empty": {}},
    }
    values.update(overrides)
    unsigned = BlobManifestV1(**values)
    return unsigned.with_signature(sign_hmac_sha256(unsigned.signing_bytes(), _KEY))


def _raw_manifest_with_user_metadata(user_metadata: dict) -> bytes:
    """Produce valid canonical bytes with a replacement user metadata map."""
    record = _manifest().to_mapping()
    record["user_metadata"] = user_metadata
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _nested_map(depth: int) -> dict:
    """Build a map whose deepest value occurs at the requested JSON depth."""
    result: dict = {"leaf": "value"}
    for _ in range(depth - 3):
        result = {"nested": result}
    return result


def test_canonical_bytes_are_stable_for_reordered_unicode_and_empty_metadata():
    """Equivalent field values always produce one byte-identical UTF-8 record."""
    first = _manifest(
        handler_metadata={"z": "last", "a": ["é", ""]},
        user_metadata={"empty": {}, "nested": {"b": 2, "a": 1}},
    )
    second = _manifest(
        handler_metadata={"a": ["é", ""], "z": "last"},
        user_metadata={"nested": {"a": 1, "b": 2}, "empty": {}},
    )

    assert first.canonical_bytes() == second.canonical_bytes()
    assert b"caf\xc3\xa9" in _manifest().canonical_bytes()
    assert BlobManifestV1.from_canonical_bytes(first.canonical_bytes()) == first


def test_manifest_raw_byte_limit_accepts_exact_boundary_and_rejects_next_byte():
    """The raw-record byte ceiling is checked before parse or payload work."""
    padding = {f"padding_{index}": "" for index in range(5)}
    raw = _raw_manifest_with_user_metadata(padding)
    remaining = MAX_MANIFEST_BYTES - len(raw)
    assert remaining > 0

    for index in padding:
        addition = min(remaining, MAX_STRING_UTF8_BYTES)
        padding[index] = "x" * addition
        remaining -= addition
    assert remaining == 0
    exact_boundary = _raw_manifest_with_user_metadata(padding)
    assert len(exact_boundary) == MAX_MANIFEST_BYTES

    assert BlobManifestV1.from_canonical_bytes(exact_boundary).user_metadata == padding
    with pytest.raises(CacheManifestIntegrityError, match="byte limit"):
        BlobManifestV1.from_canonical_bytes(exact_boundary + b" ")


@pytest.mark.parametrize("depth", [MAX_NESTING_DEPTH, MAX_NESTING_DEPTH + 1])
def test_manifest_nesting_boundary(depth: int):
    """Nested metadata accepts the exact depth limit and rejects one deeper."""
    raw = _raw_manifest_with_user_metadata(_nested_map(depth))
    if depth == MAX_NESTING_DEPTH:
        assert BlobManifestV1.from_canonical_bytes(raw).user_metadata
    else:
        with pytest.raises(CacheManifestIntegrityError, match="nesting"):
            BlobManifestV1.from_canonical_bytes(raw)


@pytest.mark.parametrize(
    ("count", "should_pass"),
    [(MAX_COLLECTION_ITEMS, True), (MAX_COLLECTION_ITEMS + 1, False)],
)
def test_manifest_collection_boundary(count: int, should_pass: bool):
    """Each map/list has an exact independently enforced item ceiling."""
    raw = _raw_manifest_with_user_metadata({"items": list(range(count))})
    if should_pass:
        assert BlobManifestV1.from_canonical_bytes(raw).user_metadata["items"]
    else:
        with pytest.raises(CacheManifestIntegrityError, match="collection"):
            BlobManifestV1.from_canonical_bytes(raw)


@pytest.mark.parametrize(
    ("value", "should_pass"),
    [
        (-(2**63), True),
        (2**63 - 1, True),
        (-(2**63) - 1, False),
        (2**63, False),
    ],
)
def test_manifest_signed_64_integer_boundary(value: int, should_pass: bool):
    """All canonical integers are signed-64 values, never booleans or floats."""
    raw = _raw_manifest_with_user_metadata({"value": value})
    if should_pass:
        assert BlobManifestV1.from_canonical_bytes(raw).user_metadata["value"] == value
    else:
        with pytest.raises(CacheManifestIntegrityError, match="signed-64"):
            BlobManifestV1.from_canonical_bytes(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"null",
        b"\xff",
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":1,"byte_size":true}',
        b'{"schema_version":1,"byte_size":1.5}',
        b'{"schema_version":1,"byte_size":NaN}',
    ],
)
def test_malformed_manifest_bytes_fail_with_typed_integrity_error(raw: bytes):
    """Ambiguous or malformed records never become ordinary cache misses."""
    with pytest.raises(CacheManifestIntegrityError):
        BlobManifestV1.from_canonical_bytes(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 2), ("payload_format_version", 2)],
)
def test_unknown_manifest_or_payload_version_fails_explicitly(field: str, value: int):
    """Schema and native-format versions are independent and never inferred."""
    record = _manifest().to_mapping()
    record[field] = value
    raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")

    with pytest.raises(CacheManifestUnsupportedVersionError):
        BlobManifestV1.from_canonical_bytes(raw)


def test_unknown_signature_algorithm_fails_with_typed_integrity_error():
    """Only the fixed HMAC-SHA256 v1 signer can authenticate a manifest."""
    record = _manifest().to_mapping()
    record["signature_algorithm"] = "other-hmac"
    raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")

    with pytest.raises(CacheManifestIntegrityError, match="signature algorithm"):
        BlobManifestV1.from_canonical_bytes(raw)


def test_signature_binds_every_critical_manifest_field():
    """Every D-09 field mutation changes the complete signed projection."""
    manifest = _manifest()
    critical_mutations = {
        "schema_version": 1,
        "key": "other-key",
        "generation": "fedcba9876543210",
        "state": "prepared",
        "locator": "/managed/other.bin",
        "handler_type": "other_handler",
        "payload_format": "other_format",
        "payload_format_version": 1,
        "digest_algorithm": "sha256",
        "digest": "b" * 64,
        "byte_size": 8,
        "created_at": "2026-08-31T00:00:00+00:00",
        "handler_metadata": {"serializer": "dill"},
        "user_metadata": {"label": "mutated"},
        "signature_algorithm": "other-hmac",
    }

    assert verify_hmac_sha256(manifest.signing_bytes(), manifest.signature, _KEY)
    for field, value in critical_mutations.items():
        signed_mapping = manifest.to_mapping(include_signature=False)
        if value == getattr(manifest, field):
            value = 2
        signed_mapping[field] = value
        mutated = json.dumps(
            signed_mapping,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        assert not verify_hmac_sha256(
            mutated, manifest.signature, _KEY
        ), field


def test_total_node_and_string_boundaries_are_independent():
    """String and aggregate-node ceilings are enforced independently of bytes."""
    accepted_string = _raw_manifest_with_user_metadata(
        {"text": "x" * MAX_STRING_UTF8_BYTES}
    )
    assert BlobManifestV1.from_canonical_bytes(accepted_string).user_metadata

    with pytest.raises(CacheManifestIntegrityError, match="string"):
        BlobManifestV1.from_canonical_bytes(
            _raw_manifest_with_user_metadata(
                {"text": "x" * (MAX_STRING_UTF8_BYTES + 1)}
            )
        )

    # The complete canonical document contributes 36 nodes before user metadata:
    # the root mapping, its 16 keys and values, and the default handler metadata.
    # The user-metadata map, four keys/lists, and their values fill the exact limit.
    final_item_count = MAX_TOTAL_NODES - (
        36 + 1 + 3 * (MAX_COLLECTION_ITEMS + 2) + 2
    )
    node_values = {
        "nodes_0": list(range(MAX_COLLECTION_ITEMS)),
        "nodes_1": list(range(MAX_COLLECTION_ITEMS)),
        "nodes_2": list(range(MAX_COLLECTION_ITEMS)),
        "nodes_3": list(range(final_item_count)),
    }
    exact_boundary = _raw_manifest_with_user_metadata(node_values)
    assert BlobManifestV1.from_canonical_bytes(exact_boundary).user_metadata

    node_values["nodes_3"].append(0)
    with pytest.raises(CacheManifestIntegrityError, match="nodes"):
        BlobManifestV1.from_canonical_bytes(_raw_manifest_with_user_metadata(node_values))


def test_builtin_writes_publish_explicit_payload_format_identity(tmp_path):
    """Successful built-in writes identify their own native payload contract."""
    config = CacheConfig(
        cache_dir=str(tmp_path),
        compression=CompressionConfig(
            pickle_compression_codec="none",
            use_blosc2_arrays=False,
        ),
    )

    array_result = ArrayHandler().put(
        np.arange(3, dtype=np.int64), tmp_path / "array", config
    )
    object_result = ObjectHandler().put({"kind": "object"}, tmp_path / "object", config)

    assert (
        array_result["payload_format"],
        array_result["payload_format_version"],
    ) == ("npz", 1)
    assert (
        object_result["payload_format"],
        object_result["payload_format_version"],
    ) == ("pickle", 1)


@pytest.mark.parametrize(
    ("handler_type", "payload_format", "payload_format_version"),
    [
        ("array", "npz", 1),
        ("array", "blosc2", 1),
        ("object", "pickle", 1),
        ("object", "dill", 1),
        ("object", "compressed_pickle", 1),
        ("object", "compressed_dill", 1),
    ],
)
def test_handler_identity_resolution_is_independent_of_payload_bytes(
    handler_type, payload_format, payload_format_version
):
    """Identity resolution trusts declarations and never opens a payload."""
    registry = HandlerRegistry()

    handler = registry.resolve_payload_contract(
        handler_type, payload_format, payload_format_version
    )

    assert handler.data_type == handler_type
    assert handler.supports_payload_contract(
        payload_format, payload_format_version
    )


@pytest.mark.parametrize(
    ("handler_type", "payload_format", "payload_format_version"),
    [
        ("array", "zip-of-pickle", 1),
        ("array", "npz", 2),
        ("object", "compressed_pickle", 2),
    ],
)
def test_unknown_handler_identity_or_independent_version_is_rejected(
    handler_type, payload_format, payload_format_version
):
    """Unknown native contracts are rejected before a future read can snapshot."""
    registry = HandlerRegistry()

    with pytest.raises(CacheManifestUnsupportedVersionError) as error:
        registry.resolve_payload_contract(
            handler_type, payload_format, payload_format_version
        )

    assert error.value.context["reason"] == "manifest_unsupported_version"


def test_native_npz_and_pickle_payloads_keep_their_library_containers(tmp_path):
    """New array and object writes stay directly consumable by NumPy and pickle."""
    config = CacheConfig(
        cache_dir=str(tmp_path),
        compression=CompressionConfig(
            pickle_compression_codec="none",
            use_blosc2_arrays=False,
        ),
    )
    array = np.arange(6, dtype=np.int32).reshape(2, 3)

    array_result = ArrayHandler().put(array, tmp_path / "native-array", config)
    array_path = Path(array_result["actual_path"])
    assert array_path.read_bytes().startswith(b"PK\x03\x04")
    with np.load(array_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["data"], array)

    object_result = ObjectHandler().put({"native": "pickle"}, tmp_path / "object", config)
    object_path = Path(object_result["actual_path"])
    assert object_path.read_bytes().startswith(b"\x80")
    assert pickle.loads(object_path.read_bytes()) == {"native": "pickle"}


def test_native_parquet_payload_keeps_its_file_signature(tmp_path):
    """The pandas handler writes a Parquet file, not a Cacheness wrapper."""
    if not PANDAS_AVAILABLE:
        pytest.skip("pandas is not enabled")

    import pandas as pd

    config = CacheConfig(cache_dir=str(tmp_path))
    frame = pd.DataFrame({"value": [1, 2]})
    result = PandasDataFrameHandler().put(frame, tmp_path / "native-frame", config)
    payload_path = Path(result["actual_path"])

    payload = payload_path.read_bytes()
    assert payload.startswith(b"PAR1")
    assert payload.endswith(b"PAR1")
    assert pd.read_parquet(payload_path).equals(frame)


def test_native_dill_payload_remains_a_dill_stream_when_selected(tmp_path):
    """Dill fallback remains a handler-owned payload rather than a wrapper."""
    dill = pytest.importorskip("dill")
    config = CacheConfig(
        cache_dir=str(tmp_path),
        compression=CompressionConfig(pickle_compression_codec="none"),
    )
    callback = lambda value: value + 1

    result = ObjectHandler().put(callback, tmp_path / "native-dill", config)
    payload = Path(result["actual_path"]).read_bytes()

    assert (result["payload_format"], result["payload_format_version"]) == ("dill", 1)
    assert payload.startswith(b"\x80")
    assert dill.loads(payload)(2) == 3


def test_legacy_blosc2_dispatch_uses_the_declared_payload_format():
    """The read-only legacy frame is selected only by its explicit identity."""
    if not BLOSC2_AVAILABLE:
        pytest.skip("blosc2 is not enabled")

    payload_path = (
        Path(__file__).parent
        / "fixtures"
        / "compat"
        / "array-raw-v035-compress"
        / "payload.b2nd"
    )
    payload = payload_path.read_bytes()
    shape_size = int.from_bytes(payload[:4], "little")
    assert payload[4 : 4 + shape_size] == b"(2, 3)"

    restored = ArrayHandler().get(payload_path, {"payload_format": "blosc2"})

    np.testing.assert_array_equal(
        restored, np.arange(6, dtype=np.int32).reshape(2, 3)
    )


def test_unsupported_handler_identity_has_no_payload_handler_event(monkeypatch):
    """An unsupported contract fails during registry resolution, before handler IO."""
    registry = HandlerRegistry()
    handler = registry.get_handler_by_type("array")
    events: list[str] = []

    def fail_if_called(*_args, **_kwargs):
        events.append("handler")
        raise AssertionError("unsupported identity must not invoke a handler")

    monkeypatch.setattr(handler, "get", fail_if_called)
    with pytest.raises(CacheManifestUnsupportedVersionError):
        registry.resolve_payload_contract("array", "future-npz", 99)

    assert events == []
