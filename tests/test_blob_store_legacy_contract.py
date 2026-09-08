"""Read-only legacy fixture recognition contract for :class:`BlobStore`."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobMigrationRequiredError
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage import legacy_manifest
from cacheness.storage.legacy_manifest import (
    LegacyManifestRecognitionError,
    recognize_legacy_fixture_tree,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "compat"
FIXTURE_IDS = (
    "array-raw-v035-compress",
    "array-raw-v037-compress2",
    "json-split-unsigned-v037",
    "json-split-signed-v038",
    "sqlite-metadata-json-v039",
    "decorator-key-v0313",
    "json-nested-v0314",
    "sqlite-columns-v0314",
)

_TRANSIENT_SQLITE_SIDECARS = {
    "metadata.sqlite3-shm",
    "metadata.sqlite3-wal",
}


def _topology(root: Path) -> StoreTopology:
    """Build a current topology only to reject a non-current fixture tree."""
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


def _copy_fixture_tree(source: Path, copied: Path, *, merge: bool = False) -> None:
    """Copy normative evidence without workspace-local SQLite journal sidecars."""

    shutil.copytree(
        source,
        copied,
        copy_function=shutil.copy2,
        dirs_exist_ok=merge,
        ignore=shutil.ignore_patterns(*_TRANSIENT_SQLITE_SIDECARS),
    )


def _tree_evidence(root: Path) -> dict[str, tuple[str, int]]:
    """Capture every fixture file's digest and nanosecond mtime."""
    evidence = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        evidence[relative] = (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
    return evidence


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_exact_legacy_fixture_identity_is_attached_in_memory_and_non_mutating(
    tmp_path: Path, fixture_id: str
) -> None:
    """Every normative Phase 1 tree has one narrow, read-only identity."""
    source = FIXTURE_ROOT / fixture_id
    copied = tmp_path / fixture_id
    _copy_fixture_tree(source, copied)
    source_before = _tree_evidence(source)
    copy_before = _tree_evidence(copied)

    identity = BlobStore.inspect_legacy_fixture_tree(copied)

    assert identity.fixture_id == fixture_id
    assert identity.source_version.startswith("0.3.")
    assert identity.schema_identity.startswith("legacy-")
    assert identity.payload_identity
    assert identity.read_only is True
    with pytest.raises(CacheBlobMigrationRequiredError):
        identity.require_explicit_migration()

    assert _tree_evidence(source) == source_before
    assert _tree_evidence(copied) == copy_before


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_blob_store_read_surfaces_report_exact_legacy_migration_without_mutation(
    tmp_path: Path, fixture_id: str
) -> None:
    """Direct read APIs never treat exact legacy evidence as an ordinary miss."""
    source = FIXTURE_ROOT / fixture_id
    copied = tmp_path / fixture_id
    _copy_fixture_tree(source, copied)
    before = _tree_evidence(copied)

    with BlobStore(_topology(copied), cache_dir=copied) as store:
        with pytest.raises(CacheBlobMigrationRequiredError):
            store.get("legacy-entry")

    assert _tree_evidence(copied) == before


def test_unknown_lookalike_is_typed_and_non_mutating(tmp_path: Path) -> None:
    """A partial split-map tree is never guessed as a legacy format."""
    source = FIXTURE_ROOT / "json-split-unsigned-v037"
    copied = tmp_path / source.name
    _copy_fixture_tree(source, copied)
    metadata_path = copied / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("file_sizes")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    os.utime(
        metadata_path,
        ns=(
            metadata_path.stat().st_atime_ns,
            source.joinpath("metadata.json").stat().st_mtime_ns,
        ),
    )
    before = _tree_evidence(copied)

    with pytest.raises(LegacyManifestRecognitionError):
        recognize_legacy_fixture_tree(copied)

    assert _tree_evidence(copied) == before


def test_malformed_signed_legacy_evidence_is_typed_and_non_mutating(tmp_path: Path) -> None:
    """Malformed signature evidence never authorizes another reader or key write."""
    source = FIXTURE_ROOT / "json-split-signed-v038"
    copied = tmp_path / source.name
    _copy_fixture_tree(source, copied)
    metadata_path = copied / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    signature_key = next(iter(metadata["entry_signature"]))
    metadata["entry_signature"][signature_key] = "g" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    os.utime(
        metadata_path,
        ns=(
            metadata_path.stat().st_atime_ns,
            source.joinpath("metadata.json").stat().st_mtime_ns,
        ),
    )
    before = _tree_evidence(copied)

    with pytest.raises(LegacyManifestRecognitionError):
        recognize_legacy_fixture_tree(copied)

    assert _tree_evidence(copied) == before


@pytest.mark.parametrize(
    ("fixture_id", "evidence_name"),
    (
        ("array-raw-v035-compress", "payload.b2nd"),
        ("json-split-unsigned-v037", "metadata.json"),
        ("json-split-signed-v038", "payload.npz"),
        ("sqlite-metadata-json-v039", "metadata.sqlite3"),
        ("json-nested-v0314", "provenance.json"),
    ),
)
def test_legacy_recognition_rejects_symlinked_evidence(
    tmp_path: Path, fixture_id: str, evidence_name: str
) -> None:
    """Every named legacy evidence kind remains contained below its root."""
    source = FIXTURE_ROOT / fixture_id
    copied = tmp_path / source.name
    _copy_fixture_tree(source, copied)
    evidence = copied / evidence_name
    outside = tmp_path / f"outside-{evidence_name}"
    outside.write_bytes(evidence.read_bytes())
    evidence.unlink()
    evidence.symlink_to(outside)

    with pytest.raises(LegacyManifestRecognitionError):
        recognize_legacy_fixture_tree(copied)


def test_signed_legacy_evidence_requires_valid_hmac_after_pinned_hash_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid-hex signature tampering is rejected by the historical verifier."""
    source = FIXTURE_ROOT / "json-split-signed-v038"
    copied = tmp_path / source.name
    _copy_fixture_tree(source, copied)
    metadata_path = copied / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    signature_key = next(iter(metadata["entry_signature"]))
    original = metadata["entry_signature"][signature_key]
    replacement = f"{'0' if original[0] != '0' else '1'}{original[1:]}"
    metadata["entry_signature"][signature_key] = replacement
    metadata["entries"][signature_key]["entry_signature"] = replacement
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    monkeypatch.setitem(
        legacy_manifest._EXACT_EVIDENCE_SHA256["json-split-signed-v038"],
        "metadata.json",
        hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(LegacyManifestRecognitionError, match="HMAC"):
        recognize_legacy_fixture_tree(copied)


@pytest.mark.parametrize(
    ("fixture_id", "evidence_name"),
    (
        ("json-split-unsigned-v037", "metadata.json"),
        ("json-split-unsigned-v037", "payload.npz"),
    ),
)
def test_legacy_recognition_requires_pinned_evidence_and_no_extra_sidecars(
    tmp_path: Path, fixture_id: str, evidence_name: str
) -> None:
    """Shape-compatible mutation and extra files cannot claim exact identity."""
    source = FIXTURE_ROOT / fixture_id
    copied = tmp_path / source.name
    _copy_fixture_tree(source, copied)
    evidence = copied / evidence_name
    raw = bytearray(evidence.read_bytes())
    raw[-1] ^= 1
    evidence.write_bytes(bytes(raw))

    with pytest.raises(LegacyManifestRecognitionError):
        recognize_legacy_fixture_tree(copied)

    _copy_fixture_tree(source, copied, merge=True)
    (copied / "unexpected.sidecar").write_bytes(b"not legacy evidence")
    with pytest.raises(LegacyManifestRecognitionError):
        recognize_legacy_fixture_tree(copied)
