"""Tests for rotate_key() API on UnifiedCache and BlobStore."""

import os
import secrets
from pathlib import Path
from typing import Any

import pytest

from cacheness.core import UnifiedCache as cacheness
from cacheness.config import (
    CacheConfig,
    CacheStorageConfig,
    CacheMetadataConfig,
    CompressionConfig,
    SecurityConfig,
)
from cacheness.error_handling import CacheSecurityError
from cacheness.interfaces import RotationResult
from cacheness.storage.blob_store import BlobStore


class HardInterruption(BaseException):
    """Simulate process death after metadata commit but before key publication."""


def _make_signed_cache(tmp_path, **security_overrides):
    """Create a cache with entry signing enabled."""
    defaults: dict[str, Any] = {
        "enable_entry_signing": True,
        "allow_unsigned_entries": True,
        "delete_invalid_signatures": False,
    }
    defaults.update(security_overrides)
    security = SecurityConfig(**defaults)
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path)),
        metadata=CacheMetadataConfig(metadata_backend="json"),
        compression=CompressionConfig(use_blosc2_arrays=False),
        security=security,
    )
    return cacheness(config)


def _generate_key_file(path):
    """Write 32 random bytes to a file and return the path."""
    path.write_bytes(secrets.token_bytes(32))
    return path


def _interrupt_key_publication(monkeypatch, active_key_path: Path) -> None:
    """Interrupt only ``<keyfile>.new`` publication; allow blob replacements."""
    staged_key_path = active_key_path.with_name(f"{active_key_path.name}.new")
    original_replace = os.replace

    def replace_or_interrupt(src, dst):
        if Path(src) == staged_key_path and Path(dst) == active_key_path:
            raise HardInterruption("process stopped before active key replacement")
        return original_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_or_interrupt)


class TestUnifiedCacheRotateKey:
    """Tests for UnifiedCache.rotate_key()."""

    def test_basic_rotation(self, tmp_path):
        """rotate_key() re-signs all entries and returns accurate counts."""
        cache = _make_signed_cache(tmp_path)
        cache.put("alpha", test_key="a")
        cache.put("beta", test_key="b")
        cache.put("gamma", test_key="c")

        new_key = _generate_key_file(tmp_path / "new_key.bin")
        result = cache.rotate_key(new_key)

        assert isinstance(result, RotationResult)
        assert result.total == 3
        assert result.re_signed == 3
        assert result.failed == 0
        assert result.skipped == 0
        assert result.failures == []

    def test_rotation_result_counts(self, tmp_path):
        """RotationResult fields are accurate for various entry counts."""
        cache = _make_signed_cache(tmp_path)
        for i in range(7):
            cache.put(f"data_{i}", test_key=f"entry_{i}")

        new_key = _generate_key_file(tmp_path / "new_key.bin")
        result = cache.rotate_key(new_key)

        assert result.total == 7
        assert result.re_signed == 7
        assert result.failed == 0

    def test_rotation_idempotent(self, tmp_path):
        """Re-running rotate_key() with same key produces consistent results."""
        cache = _make_signed_cache(tmp_path)
        cache.put("data", test_key="entry")

        new_key = _generate_key_file(tmp_path / "new_key.bin")
        result1 = cache.rotate_key(new_key)
        result2 = cache.rotate_key(new_key)

        assert result1.re_signed == 1
        assert result2.re_signed == 1
        assert result2.failed == 0

    def test_v2_to_v3_migration(self, tmp_path):
        """Entries signed with v2 (no HKDF) are migrated to v3 during rotation."""
        # Create cache with HKDF disabled → entries signed as v2
        cache = _make_signed_cache(tmp_path, use_hkdf_derivation=False)
        cache.put("old_data", test_key="v2_entry")

        # Verify entry has v2 signature
        entries = cache.metadata_backend.iter_entry_summaries()
        sig = entries[0].get("entry_signature", "")
        assert sig.startswith("v2:"), f"Expected v2 signature, got: {sig[:10]}"

        # Now rotate with HKDF enabled (default) → re-signs as v3
        cache2 = _make_signed_cache(tmp_path, use_hkdf_derivation=True)
        new_key = _generate_key_file(tmp_path / "new_key.bin")
        result = cache2.rotate_key(new_key)

        assert result.re_signed == 1

        # Verify entry now has v3 signature
        entries2 = cache2.metadata_backend.iter_entry_summaries()
        sig2 = entries2[0].get("entry_signature", "")
        assert sig2.startswith("v3:"), f"Expected v3 signature, got: {sig2[:10]}"

    def test_minimum_signature_version_blocks_v2_cache_read(self, tmp_path):
        """D-02/D-03: v2 stays readable by default but strict v3 policy rejects it."""
        legacy_cache = _make_signed_cache(tmp_path, use_hkdf_derivation=False)
        legacy_cache.put("legacy payload", test_key="legacy")
        assert legacy_cache.get(test_key="legacy") == "legacy payload"

        strict_cache = _make_signed_cache(
            tmp_path,
            use_hkdf_derivation=True,
            minimum_signature_version=3,
            delete_invalid_signatures=True,
        )

        assert strict_cache.get(test_key="legacy") is None

    def test_namespace_re_signed(self, tmp_path):
        """Namespace signature is updated during rotation."""
        cache = _make_signed_cache(tmp_path)
        cache.put("data", test_key="entry")

        # Get pre-rotation namespace signature
        ns_info_before = cache.metadata_backend.get_namespace(cache.namespace)
        sig_before = ns_info_before.signature if ns_info_before else None

        new_key = _generate_key_file(tmp_path / "new_key.bin")
        cache.rotate_key(new_key)

        # Get post-rotation namespace signature
        ns_info_after = cache.metadata_backend.get_namespace(cache.namespace)
        sig_after = ns_info_after.signature if ns_info_after else None

        assert sig_after is not None
        assert sig_after != sig_before, (
            "Namespace signature should change after rotation"
        )

    def test_entries_verify_after_rotation(self, tmp_path):
        """After rotation, all entries are readable with the new key."""
        cache = _make_signed_cache(tmp_path)
        cache.put("alpha", test_key="a")
        cache.put("beta", test_key="b")
        cache.put("gamma", test_key="c")

        new_key = _generate_key_file(tmp_path / "new_key.bin")
        cache.rotate_key(new_key)

        # All entries should be readable (signer was replaced)
        assert cache.get(test_key="a") == "alpha"
        assert cache.get(test_key="b") == "beta"
        assert cache.get(test_key="c") == "gamma"

    def test_rotation_without_signing_raises(self, tmp_path):
        """rotate_key() raises CacheSecurityError when signing is disabled."""
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            metadata=CacheMetadataConfig(metadata_backend="json"),
            compression=CompressionConfig(use_blosc2_arrays=False),
            security=SecurityConfig(enable_entry_signing=False),
        )
        cache = cacheness(config)

        new_key = _generate_key_file(tmp_path / "new_key.bin")
        with pytest.raises(CacheSecurityError, match="not enabled"):
            cache.rotate_key(new_key)

    def test_invalid_key_file_nonexistent(self, tmp_path):
        """rotate_key() raises CacheSecurityError for nonexistent key file."""
        cache = _make_signed_cache(tmp_path)

        with pytest.raises(CacheSecurityError, match="does not exist"):
            cache.rotate_key(tmp_path / "nonexistent.bin")

    def test_invalid_key_file_wrong_length(self, tmp_path):
        """rotate_key() raises CacheSecurityError for key file with wrong length."""
        cache = _make_signed_cache(tmp_path)

        bad_key = tmp_path / "bad_key.bin"
        bad_key.write_bytes(b"too_short")

        with pytest.raises(CacheSecurityError, match="Invalid key length"):
            cache.rotate_key(bad_key)

    def test_old_key_entries_handled_after_rotation(self, tmp_path):
        """After rotation, restarting cache with new key handles old entries."""
        cache = _make_signed_cache(tmp_path)
        cache.put("sensitive", test_key="important")

        # Rotate key
        new_key = _generate_key_file(tmp_path / "new_key.bin")
        cache.rotate_key(new_key)

        # Create fresh cache instance (simulates restart)
        cache2 = _make_signed_cache(tmp_path)

        # Entry should be readable — it was re-signed during rotation
        assert cache2.get(test_key="important") == "sensitive"

    def test_interrupted_rotation_keeps_active_key_and_old_entries_readable(
        self, tmp_path, monkeypatch
    ):
        """D-16/D-22: a mid-rotation metadata failure does not publish the new key."""
        cache = _make_signed_cache(
            tmp_path,
            allow_unsigned_entries=False,
            delete_invalid_signatures=False,
        )
        cache.put("alpha", test_key="a")
        cache.put("beta", test_key="b")
        assert cache.get(test_key="a") == "alpha"
        assert cache.get(test_key="b") == "beta"

        active_key = tmp_path / "cache_signing_key.bin"
        old_key_bytes = active_key.read_bytes()
        original_put_entry = cache.metadata_backend.put_entry
        failed_once = False

        def fail_once(cache_key, entry):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise RuntimeError("mid-rotation metadata failure")
            return original_put_entry(cache_key, entry)

        monkeypatch.setattr(cache.metadata_backend, "put_entry", fail_once)

        result = cache.rotate_key(_generate_key_file(tmp_path / "new_key.bin"))

        assert result.failed == 1
        assert active_key.read_bytes() == old_key_bytes
        assert cache.get(test_key="a") == "alpha"
        assert cache.get(test_key="b") == "beta"

    def test_hard_interruption_after_staged_signature_keeps_old_key_entry_readable(
        self, tmp_path, monkeypatch
    ):
        """A hard interruption after staged signatures keeps old-key reads working."""
        cache = _make_signed_cache(
            tmp_path,
            allow_unsigned_entries=False,
            delete_invalid_signatures=True,
        )
        cache.put("alpha", cache_key="hard-a")
        cache.put("beta", cache_key="hard-b")

        active_key = tmp_path / "cache_signing_key.bin"
        old_key_bytes = active_key.read_bytes()
        staged_key = active_key.with_name(f"{active_key.name}.new")
        _interrupt_key_publication(monkeypatch, active_key)

        with pytest.raises(HardInterruption):
            cache.rotate_key(_generate_key_file(tmp_path / "new_key.bin"))

        assert staged_key.exists()
        assert active_key.read_bytes() == old_key_bytes

        reopened = _make_signed_cache(
            tmp_path,
            allow_unsigned_entries=False,
            delete_invalid_signatures=True,
        )

        assert reopened.get(cache_key="hard-a") == "alpha"
        assert reopened.metadata_backend.get_entry("hard-a") is not None


class TestBlobStoreRotateKey:
    """Tests for BlobStore.rotate_key()."""

    def test_blob_store_basic_rotation(self, tmp_path):
        """BlobStore.rotate_key() re-signs all blob entries."""
        store = BlobStore(
            cache_dir=str(tmp_path / "blobs"),
            backend="json",
            enable_signing=True,
        )
        store.put("blob_a", key="a")
        store.put("blob_b", key="b")
        store.put("blob_c", key="c")

        new_key = _generate_key_file(tmp_path / "new_key.bin")
        result = store.rotate_key(new_key)

        assert isinstance(result, RotationResult)
        assert result.total == 3
        assert result.re_signed == 3
        assert result.failed == 0

    def test_blob_store_rotation_without_signing_raises(self, tmp_path):
        """BlobStore.rotate_key() raises CacheSecurityError without signing."""
        store = BlobStore(
            cache_dir=str(tmp_path / "blobs"),
            backend="json",
            enable_signing=False,
        )

        new_key = _generate_key_file(tmp_path / "new_key.bin")
        with pytest.raises(CacheSecurityError, match="not enabled"):
            store.rotate_key(new_key)

    def test_blob_store_entries_verify_after_rotation(self, tmp_path):
        """After rotation, all blobs are readable with the new key."""
        store = BlobStore(
            cache_dir=str(tmp_path / "blobs"),
            backend="json",
            enable_signing=True,
        )
        store.put("data_x", key="x")
        store.put("data_y", key="y")

        new_key = _generate_key_file(tmp_path / "new_key.bin")
        store.rotate_key(new_key)

        # Blobs should be readable (signer was replaced)
        assert store.get("x") == "data_x"
        assert store.get("y") == "data_y"

    def test_blob_store_interrupted_rotation_keeps_active_key_and_old_blobs_readable(
        self, tmp_path, monkeypatch
    ):
        """D-16/D-22: BlobStore keeps the old key active if rotation cannot finish."""
        blob_dir = tmp_path / "blobs"
        store = BlobStore(
            cache_dir=str(blob_dir),
            backend="json",
            enable_signing=True,
        )
        store.put("blob_a", key="a")
        store.put("blob_b", key="b")
        assert store.get("a") == "blob_a"
        assert store.get("b") == "blob_b"

        active_key = blob_dir / "cache_signing_key.bin"
        old_key_bytes = active_key.read_bytes()
        original_put_entry = store.backend.put_entry
        failed_once = False

        def fail_once(cache_key, entry):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise RuntimeError("mid-rotation metadata failure")
            return original_put_entry(cache_key, entry)

        monkeypatch.setattr(store.backend, "put_entry", fail_once)

        result = store.rotate_key(_generate_key_file(tmp_path / "new_blob_key.bin"))

        assert result.failed == 1
        assert active_key.read_bytes() == old_key_bytes
        assert store.get("a") == "blob_a"
        assert store.get("b") == "blob_b"

    def test_blob_store_hard_interruption_after_staged_signature_keeps_old_key_blob_readable(
        self, tmp_path, monkeypatch
    ):
        """BlobStore uses staged signatures after hard interruption before key publish."""
        blob_dir = tmp_path / "blobs"
        store = BlobStore(
            cache_dir=str(blob_dir),
            backend="json",
            enable_signing=True,
            config=CacheConfig(
                cache_dir=blob_dir,
                security=SecurityConfig(
                    enable_entry_signing=True,
                    allow_unsigned_entries=False,
                    delete_invalid_signatures=True,
                ),
            ),
        )
        store.put("blob_a", key="hard-a")
        store.put("blob_b", key="hard-b")

        active_key = blob_dir / "cache_signing_key.bin"
        old_key_bytes = active_key.read_bytes()
        staged_key = active_key.with_name(f"{active_key.name}.new")
        _interrupt_key_publication(monkeypatch, active_key)

        with pytest.raises(HardInterruption):
            store.rotate_key(_generate_key_file(tmp_path / "new_blob_key.bin"))

        assert staged_key.exists()
        assert active_key.read_bytes() == old_key_bytes

        reopened = BlobStore(
            cache_dir=str(blob_dir),
            backend="json",
            enable_signing=True,
            config=CacheConfig(
                cache_dir=blob_dir,
                security=SecurityConfig(
                    enable_entry_signing=True,
                    allow_unsigned_entries=False,
                    delete_invalid_signatures=True,
                ),
            ),
        )

        assert reopened.get("hard-a") == "blob_a"
        assert reopened.get_metadata("hard-a") is not None
