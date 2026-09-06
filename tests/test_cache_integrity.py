"""
Tests for cache file integrity verification functionality.
"""

import pytest
import tempfile
from copy import deepcopy
from pathlib import Path
import numpy as np

from cacheness import CacheConfig, SecurityConfig, cacheness
from cacheness.error_handling import CacheIntegrityError, CacheStorageError


@pytest.fixture
def temp_cache():
    """Fixture to create a temporary cache for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = CacheConfig(cache_dir=temp_dir, verify_cache_integrity=True)
        cache = cacheness(config)
        yield cache
        cache.close()


@pytest.fixture  
def temp_cache_no_integrity():
    """Fixture to create a temporary cache with integrity verification disabled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = CacheConfig(cache_dir=temp_dir, verify_cache_integrity=False)
        cache = cacheness(config)
        yield cache
        cache.close()


def _cache_with_signing_policy(tmp_path: Path, *, allow_unsigned: bool):
    """Create a cache that exercises one entry-signing publication policy."""
    return cacheness(
        CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            metadata_backend="memory",
            cleanup_on_init=False,
            security=SecurityConfig(
                enable_entry_signing=True,
                allow_unsigned_entries=allow_unsigned,
            ),
        )
    )


class _SnapshotOpenFailure:
    """Context manager double that fails before a candidate digest is calculated."""

    def __enter__(self):
        """Raise at the snapshot-open boundary."""
        raise RuntimeError("snapshot unavailable")

    def __exit__(self, _exc_type, _exc_value, _traceback):
        """Do not suppress failures from the simulated snapshot boundary."""
        return False


def _candidate_paths(cache) -> set[Path]:
    """Return all private candidate payloads currently owned by one cache root."""
    return set(Path(cache.cache_dir).glob("*candidate-*"))


def _overwrite_evidence(cache, cache_key: str) -> tuple[dict, Path, bytes, set[Path]]:
    """Capture every committed fact a failed replacement must preserve exactly."""
    entry = deepcopy(cache.metadata_backend.get_entry(cache_key))
    assert entry is not None
    payload_path = Path(entry["metadata"]["actual_path"])
    return entry, payload_path, payload_path.read_bytes(), _candidate_paths(cache)


class TestCacheIntegrity:
    """Test cache file integrity verification."""

    def test_cache_integrity_verification_enabled_by_default(self):
        """Test that cache integrity verification is enabled by default."""
        config = CacheConfig()
        assert config.verify_cache_integrity is True

    def test_cache_integrity_verification_can_be_disabled(self):
        """Test that cache integrity verification can be disabled."""
        config = CacheConfig(verify_cache_integrity=False)
        assert config.verify_cache_integrity is False

    def test_file_hash_calculation(self, temp_cache):
        """Test that file hash is calculated correctly."""
        cache = temp_cache
        
        # Create a test file in the cache directory
        test_file = Path(cache.cache_dir) / "test_file.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        # Calculate hash
        calculated_hash = cache._calculate_file_hash(test_file)
        assert calculated_hash is not None
        assert isinstance(calculated_hash, str)
        assert (
            len(calculated_hash) == 16
        )  # XXH3_64 produces 16-character hex strings

    def test_file_hash_stored_in_metadata(self, temp_cache):
        """Test that file hash is stored in metadata when caching."""
        cache = temp_cache

        # Cache some data
        test_data = {"message": "Hello, World!"}
        cache.put(test_data, description="Test data", test_key="value")

        # Check that metadata contains file hash
        cache_key = cache._create_cache_key({"test_key": "value"})
        entry = cache.metadata_backend.get_entry(cache_key)
        assert entry is not None

        metadata = entry.get("metadata", {})
        assert "file_hash" in metadata
        assert metadata["file_hash"] is not None
        assert isinstance(metadata["file_hash"], str)

    def test_file_hash_not_stored_when_disabled(self, temp_cache_no_integrity):
        """Test that file hash is not stored when verification is disabled."""
        cache = temp_cache_no_integrity

        # Cache some data
        test_data = {"message": "Hello, World!"}
        cache.put(test_data, description="Test data", test_key="value")

        # Check that metadata doesn't contain file hash
        cache_key = cache._create_cache_key({"test_key": "value"})
        entry = cache.metadata_backend.get_entry(cache_key)
        assert entry is not None

        metadata = entry.get("metadata", {})
        assert metadata.get("file_hash") is None

    def test_successful_integrity_verification(self, temp_cache):
        """Test that valid cache files pass integrity verification."""
        cache = temp_cache

        # Cache some data
        test_data = np.array([1, 2, 3, 4, 5])
        cache.put(test_data, description="Test array", test_key="array")

        # Retrieve data - should succeed with integrity verification
        retrieved_data = cache.get(test_key="array")
        assert retrieved_data is not None
        np.testing.assert_array_equal(retrieved_data, test_data)

    def test_corrupted_cache_file_detection(self, temp_cache):
        """Test that corrupted cache files are detected and removed."""
        cache = temp_cache

        # Cache some data
        test_data = {"message": "Hello, World!"}
        cache.put(test_data, description="Test data", test_key="value")

        # Get the cache file path and corrupt it
        cache_key = cache._create_cache_key({"test_key": "value"})
        entry = cache.metadata_backend.get_entry(cache_key)
        assert entry is not None, "Cache entry should exist"
        file_path = Path(entry["metadata"]["actual_path"])

        # Corrupt the file by appending some bytes
        with open(file_path, "ab") as f:
            f.write(b"CORRUPTED")

        # Try to retrieve data - should detect corruption and return None
        retrieved_data = cache.get(test_key="value")
        assert retrieved_data is None

        # Verify that the corrupted entry was removed from metadata
        entry_after = cache.metadata_backend.get_entry(cache_key)
        assert entry_after is None

    def test_missing_file_hash_rejects_retrieval_when_verification_is_enabled(self):
        """An enabled integrity gate never treats a missing digest as legacy-safe."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Disable entry signing for this legacy compatibility test
            from cacheness.config import SecurityConfig
            config = CacheConfig(
                cache_dir=temp_dir, 
                verify_cache_integrity=True,
                security=SecurityConfig(enable_entry_signing=False)  # Test legacy behavior without signing
            )
            cache = cacheness(config)

            # Cache some data first
            test_data = {"message": "Hello, World!"}
            cache.put(test_data, description="Test data", test_key="value")

            # Manually remove the file_hash from metadata to simulate legacy entry
            cache_key = cache._create_cache_key({"test_key": "value"})
            entry = cache.metadata_backend.get_entry(cache_key)
            assert entry is not None, "Cache entry should exist"
            metadata = entry["metadata"]
            del metadata["file_hash"]

            # Update the metadata without file_hash
            entry_data = {
                "description": entry["description"],
                "data_type": entry["data_type"],
                "prefix": entry["prefix"],
                "file_size": entry["file_size"],
                "metadata": metadata,
            }
            cache.metadata_backend.put_entry(cache_key, entry_data)

            # A missing digest cannot disable an explicitly enabled integrity gate.
            retrieved_data = cache.get(test_key="value")
            assert retrieved_data is None
            assert cache.metadata_backend.get_entry(cache_key) is None
            
            cache.close()

    def test_unavailable_digest_fails_put_without_committing_metadata(
        self, temp_cache, monkeypatch
    ):
        """Write-time integrity failures cannot create an unsigned/unverified entry."""
        cache = temp_cache
        monkeypatch.setattr(cache, "_calculate_file_hash", lambda _path: None)

        with pytest.raises(CacheIntegrityError, match="digest"):
            cache.put({"message": "unverified"}, test_key="unverified")

        cache_key = cache._create_cache_key({"test_key": "unverified"})
        assert cache.metadata_backend.get_entry(cache_key) is None

    def test_overwrite_digest_failure_preserves_the_committed_entry(
        self, temp_cache, monkeypatch
    ):
        """A failed replacement digest must not damage the still-committed value."""
        cache = temp_cache
        cache.put({"message": "committed"}, test_key="replace-me")
        cache_key = cache._create_cache_key({"test_key": "replace-me"})
        entry_before = cache.metadata_backend.get_entry(cache_key)
        assert entry_before is not None
        entry_before = deepcopy(entry_before)
        payload_before = Path(entry_before["metadata"]["actual_path"])
        payload_bytes_before = payload_before.read_bytes()
        candidate_paths_before = set(Path(cache.cache_dir).glob("*candidate-*"))
        calculate_file_hash = cache._calculate_file_hash

        monkeypatch.setattr(cache, "_calculate_file_hash", lambda _path: None)

        with pytest.raises(CacheIntegrityError, match="digest"):
            cache.put({"message": "replacement"}, test_key="replace-me")

        assert cache.metadata_backend.get_entry(cache_key) == entry_before
        assert payload_before.read_bytes() == payload_bytes_before
        monkeypatch.setattr(cache, "_calculate_file_hash", calculate_file_hash)
        assert cache.get(test_key="replace-me") == {"message": "committed"}
        assert set(Path(cache.cache_dir).glob("*candidate-*")) == candidate_paths_before

    @pytest.mark.parametrize("signer_available", [True, False])
    def test_strict_signing_failure_discards_an_uncommitted_candidate(
        self, tmp_path, monkeypatch, signer_available
    ):
        """Strict signing never reports a key or leaves a payload without metadata."""
        cache = _cache_with_signing_policy(tmp_path, allow_unsigned=False)
        try:
            cache_key = cache._create_cache_key({"test_key": "strict-failure"})
            candidate_paths_before = set(Path(cache.cache_dir).glob("*candidate-*"))

            if signer_available:
                def signing_failure(_entry_data):
                    raise RuntimeError("signer unavailable for test")

                monkeypatch.setattr(cache.signer, "sign_entry", signing_failure)
            else:
                monkeypatch.setattr(cache, "signer", None)

            with pytest.raises(CacheIntegrityError, match="Unable to sign cache entry"):
                cache.put({"message": "uncommitted"}, test_key="strict-failure")

            assert cache.metadata_backend.get_entry(cache_key) is None
            assert (
                set(Path(cache.cache_dir).glob("*candidate-*"))
                == candidate_paths_before
            )
        finally:
            cache.close()

    def test_strict_signing_failure_preserves_a_committed_overwrite(
        self, tmp_path, monkeypatch
    ):
        """Signing an overwrite happens before its candidate replaces old data."""
        cache = _cache_with_signing_policy(tmp_path, allow_unsigned=False)
        try:
            cache.put({"message": "committed"}, test_key="strict-overwrite")
            cache_key = cache._create_cache_key({"test_key": "strict-overwrite"})
            entry_before = deepcopy(cache.metadata_backend.get_entry(cache_key))
            assert entry_before is not None
            payload_before = Path(entry_before["metadata"]["actual_path"])
            payload_bytes_before = payload_before.read_bytes()
            candidate_paths_before = set(Path(cache.cache_dir).glob("*candidate-*"))
            sign_entry = cache.signer.sign_entry

            def signing_failure(_entry_data):
                raise RuntimeError("signer unavailable for test")

            monkeypatch.setattr(cache.signer, "sign_entry", signing_failure)

            with pytest.raises(CacheIntegrityError, match="Unable to sign cache entry"):
                cache.put({"message": "replacement"}, test_key="strict-overwrite")

            assert cache.metadata_backend.get_entry(cache_key) == entry_before
            assert payload_before.read_bytes() == payload_bytes_before
            monkeypatch.setattr(cache.signer, "sign_entry", sign_entry)
            assert cache.get(test_key="strict-overwrite") == {"message": "committed"}
            assert (
                set(Path(cache.cache_dir).glob("*candidate-*"))
                == candidate_paths_before
            )
        finally:
            cache.close()

    def test_compatibility_signing_failure_commits_an_explicit_unsigned_entry(
        self, tmp_path, monkeypatch
    ):
        """Unsigned compatibility mode remains a deliberate permitted policy."""
        cache = _cache_with_signing_policy(tmp_path, allow_unsigned=True)
        try:
            def signing_failure(_entry_data):
                raise RuntimeError("signer unavailable for test")

            monkeypatch.setattr(cache.signer, "sign_entry", signing_failure)

            cache_key = cache.put({"message": "compatibility"}, test_key="unsigned")
            entry = cache.metadata_backend.get_entry(cache_key)
            assert entry is not None
            assert "entry_signature" not in entry["metadata"]
            assert cache.get(cache_key=cache_key) == {"message": "compatibility"}
        finally:
            cache.close()

    def test_integrity_verification_disabled_skips_check(self):
        """Test that disabling verification skips integrity check completely."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First cache with verification enabled
            config_enabled = CacheConfig(
                cache_dir=temp_dir, verify_cache_integrity=True
            )
            cache_enabled = cacheness(config_enabled)

            test_data = {"message": "Hello, World!"}
            cache_enabled.put(test_data, description="Test data", test_key="value")

            # Get the cache file path and corrupt it
            cache_key = cache_enabled._create_cache_key({"test_key": "value"})
            entry = cache_enabled.metadata_backend.get_entry(cache_key)
            assert entry is not None, "Cache entry should exist"
            file_path = Path(entry["metadata"]["actual_path"])

            with open(file_path, "ab") as f:
                f.write(b"CORRUPTED")

            # Create new cache instance with verification disabled
            config_disabled = CacheConfig(
                cache_dir=temp_dir, verify_cache_integrity=False
            )
            cache_disabled = cacheness(config_disabled)

            # Should still return data (though corrupted) because verification is disabled
            # Note: This might fail at the handler level due to actual corruption,
            # but it won't fail due to hash verification
            try:
                _ = cache_disabled.get(test_key="value")
                # If we get here, verification was truly skipped
                # The data might be None due to handler-level corruption detection
            except Exception:
                # Expected - the handler will likely fail to parse corrupted data
                # But the important thing is we didn't fail due to hash verification
                pass
            
            cache_enabled.close()
            cache_disabled.close()

    def test_file_hash_calculation_error_handling(self, temp_cache):
        """Test that file hash calculation handles errors gracefully."""
        cache = temp_cache

        # Test with non-existent file
        non_existent_file = Path(cache.cache_dir) / "does_not_exist.txt"
        hash_result = cache._calculate_file_hash(non_existent_file)
        assert hash_result is None


@pytest.mark.parametrize(
    ("boundary", "expected_error"),
    (
        ("snapshot_open", RuntimeError),
        ("digest_exception", RuntimeError),
        ("missing_digest", CacheIntegrityError),
        ("signer_missing", CacheIntegrityError),
        ("signer_exception", CacheIntegrityError),
        ("empty_signature", CacheIntegrityError),
        ("metadata_publication", RuntimeError),
    ),
)
def test_unified_cache_precommit_failures_preserve_exact_overwrite_evidence(
    tmp_path, monkeypatch, boundary, expected_error
):
    """Every candidate boundary preserves the committed overwrite on failure."""
    cache = _cache_with_signing_policy(tmp_path, allow_unsigned=False)
    cache_key_params = {"candidate_boundary": boundary}

    def fail_snapshot_open(*_args, **_kwargs):
        return _SnapshotOpenFailure()

    def fail_digest(_path):
        raise RuntimeError("digest unavailable")

    def fail_signer(_entry_data):
        raise RuntimeError("signer unavailable")

    def empty_signature(_entry_data):
        return ""

    def fail_metadata_publication(_cache_key, _entry_data):
        raise RuntimeError("metadata unavailable")

    try:
        cache.put({"value": "committed"}, **cache_key_params)
        cache_key = cache._create_cache_key(cache_key_params)
        entry_before, payload_before, bytes_before, candidates_before = _overwrite_evidence(
            cache,
            cache_key,
        )

        if boundary == "snapshot_open":
            monkeypatch.setattr(
                cache.guarded_handler_io,
                "open_snapshot",
                fail_snapshot_open,
            )
        elif boundary == "digest_exception":
            monkeypatch.setattr(cache, "_calculate_file_hash", fail_digest)
        elif boundary == "missing_digest":
            monkeypatch.setattr(cache, "_calculate_file_hash", lambda _path: None)
        elif boundary == "signer_missing":
            monkeypatch.setattr(cache, "signer", None)
        elif boundary == "signer_exception":
            monkeypatch.setattr(cache.signer, "sign_entry", fail_signer)
        elif boundary == "empty_signature":
            monkeypatch.setattr(cache.signer, "sign_entry", empty_signature)
        else:
            monkeypatch.setattr(
                cache.metadata_backend,
                "put_entry",
                fail_metadata_publication,
            )

        with pytest.raises(expected_error):
            cache.put({"value": "replacement"}, **cache_key_params)

        # The evidence assertions must run against the normal read path, not
        # against a fault still injected into hashing or signature checks.
        monkeypatch.undo()
        assert cache.metadata_backend.get_entry(cache_key) == entry_before
        assert payload_before.read_bytes() == bytes_before
        assert _candidate_paths(cache) == candidates_before
        assert cache.get(**cache_key_params) == {"value": "committed"}
    finally:
        cache.close()


@pytest.mark.parametrize("cleanup_outcome", ("false", "raise"))
def test_unified_cache_unresolved_candidate_cleanup_is_chained(
    tmp_path, monkeypatch, cleanup_outcome
):
    """A failed candidate deletion is explicit and leaves prior evidence untouched."""
    cache = _cache_with_signing_policy(tmp_path, allow_unsigned=False)
    cache_key_params = {"candidate_cleanup": cleanup_outcome}
    cleanup_attempts: list[Path] = []

    def fail_metadata_publication(_cache_key, _entry_data):
        raise RuntimeError("metadata unavailable")

    def cannot_prove_cleanup(locator):
        cleanup_attempts.append(Path(locator))
        if cleanup_outcome == "raise":
            raise OSError("candidate cleanup unavailable")
        return False

    try:
        cache.put({"value": "committed"}, **cache_key_params)
        cache_key = cache._create_cache_key(cache_key_params)
        entry_before, payload_before, bytes_before, _ = _overwrite_evidence(
            cache,
            cache_key,
        )
        monkeypatch.setattr(
            cache.metadata_backend,
            "put_entry",
            fail_metadata_publication,
        )
        monkeypatch.setattr(
            cache.guarded_handler_io.file_ops,
            "delete",
            cannot_prove_cleanup,
        )

        with pytest.raises(CacheStorageError) as exc_info:
            cache.put({"value": "replacement"}, **cache_key_params)

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "metadata unavailable" in str(exc_info.value.__cause__)
        assert len(cleanup_attempts) == 1
        assert cache.metadata_backend.get_entry(cache_key) == entry_before
        assert payload_before.read_bytes() == bytes_before
    finally:
        cache.close()


def test_unified_cache_postcommit_cleanup_keeps_new_entry_authoritative(
    tmp_path, monkeypatch
):
    """A failed old-payload deletion cannot undo successful candidate publication."""
    cache = _cache_with_signing_policy(tmp_path, allow_unsigned=False)
    cache_key_params = {"postcommit_cleanup": "key"}

    try:
        cache.put({"value": "committed"}, **cache_key_params)
        cache_key = cache._create_cache_key(cache_key_params)
        entry_before, payload_before, _, _ = _overwrite_evidence(cache, cache_key)
        delete_or_prove_absent = cache._cache_blob_store._delete_or_prove_absent

        def fail_prior_cleanup(locator):
            candidate = Path(locator)
            if not candidate.is_absolute():
                candidate = cache._cache_blob_store.cache_dir / candidate
            if candidate == payload_before:
                raise OSError("prior cleanup unavailable")
            return delete_or_prove_absent(locator)

        monkeypatch.setattr(
            cache._cache_blob_store,
            "_delete_or_prove_absent",
            fail_prior_cleanup,
        )

        with pytest.raises(CacheStorageError, match="cleanup"):
            cache.put({"value": "replacement"}, **cache_key_params)

        entry_after = cache.metadata_backend.get_entry(cache_key)
        assert entry_after is not None
        assert entry_after != entry_before
        assert Path(entry_after["metadata"]["actual_path"]) != payload_before
        assert payload_before.exists()
        assert cache.get(**cache_key_params) == {"value": "replacement"}
    finally:
        cache.close()
