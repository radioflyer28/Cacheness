"""Fault injection at the authoritative storage boundary, never a derived view."""

from dataclasses import replace
import hashlib

import pytest

from cacheness.storage.manifest import BlobManifestV1
from cacheness.storage.integrity import sign_hmac_sha256


@pytest.fixture
def rewrite_authority_manifest(monkeypatch):
    """Model authenticated but invalid persisted fields without changing payloads.

    Re-signing is deliberate: tests can exercise containment or cache signing
    policy beyond the outer manifest authentication gate. The production writer
    does not expose this ability. Only this test's authority observations change.
    """
    def rewrite(store, key, change):
        authority = store.lifecycle_authority
        original_read = authority.read_entry
        original_list = authority.list_entries
        entry = original_read(key)
        assert entry is not None
        fields = BlobManifestV1.from_canonical_bytes(entry.manifest).to_mapping()
        fields.pop("signature")
        change(fields)
        manifest = BlobManifestV1(**fields)
        raw = manifest.with_signature(
            sign_hmac_sha256(manifest.signing_bytes(), store._authority_manifest_key())
        ).canonical_bytes()
        changed = replace(
            entry, locator=fields["locator"], manifest=raw,
            expectation=replace(
                entry.expectation, manifest_digest=hashlib.sha256(raw).hexdigest()
            ),
        )
        monkeypatch.setattr(
            authority, "read_entry",
            lambda requested: changed if requested == key else original_read(requested),
        )
        monkeypatch.setattr(
            authority, "list_entries",
            lambda: tuple(changed if item.key == key else item for item in original_list()),
        )
        return changed
    return rewrite
