"""Read-only legacy fixture recognition contract for :class:`BlobStore`."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobMigrationRequiredError
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
def test_exact_legacy_fixture_identity_is_attached_in_memory_without_mutation(
    tmp_path: Path, fixture_id: str
) -> None:
    """Every normative Phase 1 tree has one narrow, read-only identity."""
    source = FIXTURE_ROOT / fixture_id
    copied = tmp_path / fixture_id
    shutil.copytree(source, copied, copy_function=shutil.copy2)
    source_before = _tree_evidence(source)
    copy_before = _tree_evidence(copied)

    identity = recognize_legacy_fixture_tree(copied)

    assert identity.fixture_id == fixture_id
    assert identity.source_version.startswith("0.3.")
    assert identity.schema_identity.startswith("legacy-")
    assert identity.payload_identity
    assert identity.read_only is True
    with pytest.raises(CacheBlobMigrationRequiredError):
        identity.require_explicit_migration()

    assert _tree_evidence(source) == source_before
    assert _tree_evidence(copied) == copy_before


def test_unknown_lookalike_is_typed_and_never_falls_back_or_mutates(tmp_path: Path) -> None:
    """A similar tree with extra evidence is never guessed as a legacy format."""
    source = FIXTURE_ROOT / "json-split-unsigned-v037"
    copied = tmp_path / source.name
    shutil.copytree(source, copied, copy_function=shutil.copy2)
    (copied / "unexpected.json").write_text("{}", encoding="utf-8")
    before = _tree_evidence(copied)

    with pytest.raises(LegacyManifestRecognitionError):
        recognize_legacy_fixture_tree(copied)

    assert _tree_evidence(copied) == before


def test_malformed_signed_legacy_evidence_is_typed_and_non_mutating(tmp_path: Path) -> None:
    """Malformed signature evidence never authorizes another reader or key write."""
    source = FIXTURE_ROOT / "json-split-signed-v038"
    copied = tmp_path / source.name
    shutil.copytree(source, copied, copy_function=shutil.copy2)
    metadata_path = copied / "metadata.json"
    payload = metadata_path.read_text(encoding="utf-8").replace(
        '"entry_signature": "', '"entry_signature": "not-a-valid-signature', 1
    )
    metadata_path.write_text(payload, encoding="utf-8")
    os.utime(metadata_path, ns=(metadata_path.stat().st_atime_ns, source.joinpath("metadata.json").stat().st_mtime_ns))
    before = _tree_evidence(copied)

    with pytest.raises(LegacyManifestRecognitionError):
        recognize_legacy_fixture_tree(copied)

    assert _tree_evidence(copied) == before
