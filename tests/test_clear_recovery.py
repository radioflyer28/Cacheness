"""LifecycleAuthority clear contracts for canonical BlobStore stores."""

from cacheness.storage.blob_store import BlobStore


def test_authority_clear_removes_only_committed_entries(tmp_path):
    """Clear revokes every committed authority entry and its payload."""
    store = BlobStore(tmp_path / "authority-clear", backend="json")
    try:
        first = store.put("first", key="first")
        second = store.put("second", key="second")

        assert store.clear() == 2
        assert store.get(first) is None
        assert store.get(second) is None
        assert store.get_metadata(first) is None
        assert store.list() == []
        assert store.lifecycle_authority.read_entry(first) is None
        assert store.lifecycle_authority.read_entry(second) is None
    finally:
        store.close()


def test_authority_clear_ignores_retired_journal_evidence(tmp_path):
    """Legacy clear journals cannot authorize mutations of canonical entries."""
    root = tmp_path / "authority-clear-legacy-evidence"
    store = BlobStore(root, backend="json")
    journal_path = root / ".cacheness-clear-journal-v1.json"
    evidence = b'{"owner":"forged","state":"committed"}'
    try:
        key = store.put("preserved", key="preserved")
        journal_path.write_bytes(evidence)
    finally:
        store.close()

    reopened = BlobStore(root, backend="json")
    try:
        assert reopened.get(key) == "preserved"
        assert reopened.reconcile().applied is False
        assert journal_path.read_bytes() == evidence
        assert reopened.clear() == 1
        assert journal_path.read_bytes() == evidence
    finally:
        reopened.close()
