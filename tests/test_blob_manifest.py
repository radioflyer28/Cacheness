"""Current format-2 canonical descriptor and native payload regressions."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest

from cacheness.config import CacheConfig, CompressionConfig
from cacheness.error_handling import CacheManifestIntegrityError
from cacheness.handlers import (
    ArrayHandler,
    HandlerRegistry,
    ObjectHandler,
    PANDAS_AVAILABLE,
    PandasDataFrameHandler,
)
from cacheness.storage import BlobReceipt, BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.manifest import (
    BlobManifest,
    MigrationOrRebuildRequired,
    StoreVersionDimensions,
    sign_current_manifest,
    verify_current_manifest,
)


_KEY = b"canonical-manifest-test-key-0001"


def _topology(root: Path) -> StoreTopology:
    """Create the qualified local filesystem/SQLite topology."""
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


def _manifest(**overrides: object) -> BlobManifest:
    """Build one complete format-2 descriptor for codec assertions."""
    values: dict[str, object] = {
        "versions": StoreVersionDimensions(),
        "key": "tenant/asset",
        "generation": "0123456789abcdef",
        "locator": "generations/tenant/asset.bin",
        "handler_type": "object",
        "payload_format": "pickle",
        "digest": "a" * 64,
        "byte_size": 7,
        "created_at": "2026-09-08T00:00:00+00:00",
        "catalog_schema_id": "application",
        "catalog_schema_revision": 1,
        "catalog_schema_fingerprint": hashlib.sha256(b"application").hexdigest(),
        "catalog_values": {"label": "caf\u00e9"},
        "catalog_presence": ("label",),
        "user_metadata": {"label": "caf\u00e9"},
        "handler_metadata": {"serializer": "pickle"},
    }
    values.update(overrides)
    return sign_current_manifest(BlobManifest(**values), _KEY)


def test_current_descriptor_is_stable_and_authenticates_all_version_dimensions() -> None:
    """Canonical format-2 bytes include independent store and payload versions."""
    first = _manifest(catalog_values={"a": 1, "z": "last"}, catalog_presence=("a", "z"))
    second = _manifest(catalog_values={"z": "last", "a": 1}, catalog_presence=("a", "z"))

    assert first.canonical_bytes() == second.canonical_bytes()
    assert BlobManifest.from_canonical_bytes(first.canonical_bytes()) == first
    verify_current_manifest(first, _KEY)

    altered = BlobManifest.from_mapping(
        {**first.to_mapping(), "generation": "fedcba9876543210"}
    )
    with pytest.raises(CacheManifestIntegrityError, match="authentication"):
        verify_current_manifest(altered, _KEY)


def test_pre_format_two_descriptor_is_rejected_without_compatibility_reading() -> None:
    """A schema-1 shape is migration evidence, never a live direct-store input."""
    legacy = b'{"schema_version":1,"key":"legacy"}'

    with pytest.raises(MigrationOrRebuildRequired):
        BlobManifest.from_canonical_bytes(legacy)


def test_composed_store_commits_a_signed_descriptor_and_frozen_receipt(tmp_path: Path) -> None:
    """Direct lifecycle writes expose a receipt for the exact committed generation."""
    root = tmp_path / "canonical-store"
    store = BlobStore(_topology(root), cache_dir=root)
    try:
        receipt = store.put_entry(
            {"answer": 42}, key="current", metadata={"label": "answer"}
        )
        committed = store.lifecycle_authority.read_entry(receipt.key)

        assert isinstance(receipt, BlobReceipt)
        assert committed is not None
        descriptor = BlobManifest.from_canonical_bytes(committed.manifest)
        verify_current_manifest(descriptor, store._authority_manifest_key())
        assert (descriptor.key, descriptor.generation, descriptor.locator) == (
            receipt.key,
            receipt.generation,
            receipt.locator,
        )
        assert store.get(receipt.key) == {"answer": 42}
    finally:
        store.close()


def test_builtin_writes_publish_explicit_payload_format_identity(tmp_path: Path) -> None:
    """Native handlers keep their format identity independent of store format."""
    config = CacheConfig(
        cache_dir=str(tmp_path),
        compression=CompressionConfig(
            pickle_compression_codec="none",
            use_blosc2_arrays=False,
        ),
    )

    array_result = ArrayHandler().put(np.arange(3, dtype=np.int64), tmp_path / "array", config)
    object_result = ObjectHandler().put({"kind": "object"}, tmp_path / "object", config)

    assert (array_result["payload_format"], array_result["payload_format_version"]) == (
        "npz",
        1,
    )
    assert (object_result["payload_format"], object_result["payload_format_version"]) == (
        "pickle",
        1,
    )


def test_native_payloads_keep_their_library_containers(tmp_path: Path) -> None:
    """Descriptors do not wrap native NumPy and pickle payload bytes."""
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
    assert pickle.loads(Path(object_result["actual_path"]).read_bytes()) == {
        "native": "pickle"
    }


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas is not enabled")
def test_native_parquet_payload_keeps_its_file_signature(tmp_path: Path) -> None:
    """Pandas data remains handler-owned Parquet bytes."""
    import pandas as pd

    frame = pd.DataFrame({"value": [1, 2]})
    result = PandasDataFrameHandler().put(
        frame, tmp_path / "native-frame", CacheConfig(cache_dir=str(tmp_path))
    )
    payload = Path(result["actual_path"]).read_bytes()

    assert payload.startswith(b"PAR1")
    assert payload.endswith(b"PAR1")
    assert pd.read_parquet(result["actual_path"]).equals(frame)


def test_unknown_handler_identity_is_rejected_before_payload_handler_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported payload contracts fail before a handler opens attacker-controlled bytes."""
    registry = HandlerRegistry()
    handler = registry.get_handler_by_type("array")
    events: list[str] = []

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        events.append("handler")
        raise AssertionError("unsupported identity must not invoke a handler")

    monkeypatch.setattr(handler, "get", fail_if_called)
    with pytest.raises(Exception):
        registry.resolve_payload_contract("array", "future-npz", 99)

    assert events == []
