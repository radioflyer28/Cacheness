"""
Tests for HKDF-SHA256 per-namespace key derivation (Phase 16, SEC-01).

Validates:
1. HKDF function produces deterministic, unique 32-byte keys
2. v3 entry signatures use derived keys, v2 entries still verify
3. ns2 namespace signatures use derived keys, ns1 entries still verify
4. SecurityConfig integration and create_cache_signer factory wiring
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path

from cacheness.security import CacheEntrySigner, _hkdf_sha256, create_cache_signer
from cacheness.config import SecurityConfig


# ---------------------------------------------------------------------------
# HKDF function unit tests
# ---------------------------------------------------------------------------


class TestHkdfFunction:
    """Unit tests for the _hkdf_sha256 derivation function."""

    def test_deterministic(self):
        """Same inputs always produce the same output."""
        ikm = b"master-key-material-32-bytes!!!!!"
        info = b"cacheness-ns-v1:default"
        result1 = _hkdf_sha256(ikm, info)
        result2 = _hkdf_sha256(ikm, info)
        assert result1 == result2

    def test_different_info_different_keys(self):
        """Different namespace IDs produce different derived keys."""
        ikm = b"master-key-material-32-bytes!!!!!"
        key_a = _hkdf_sha256(ikm, b"cacheness-ns-v1:namespace_a")
        key_b = _hkdf_sha256(ikm, b"cacheness-ns-v1:namespace_b")
        assert key_a != key_b

    def test_output_length_32(self):
        """Output is exactly 32 bytes."""
        ikm = b"master-key-material-32-bytes!!!!!"
        result = _hkdf_sha256(ikm, b"test-info")
        assert len(result) == 32
        assert isinstance(result, bytes)

    def test_different_master_key_different_output(self):
        """Different master keys produce different derived keys."""
        info = b"cacheness-ns-v1:default"
        key_a = _hkdf_sha256(b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", info)
        key_b = _hkdf_sha256(b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", info)
        assert key_a != key_b


# ---------------------------------------------------------------------------
# Entry signature versioning tests
# ---------------------------------------------------------------------------


class TestHkdfSignatureVersioning:
    """Tests for v3 entry signatures with HKDF-derived keys."""

    SAMPLE_ENTRY = {
        "cache_key": "test_key_123",
        "data_type": "object",
        "file_size": 1024,
        "file_hash": "abc123def456",
        "object_type": "dict",
        "storage_format": "pickle",
        "serializer": "compress_pickle",
        "compression_codec": "lz4",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }

    def test_v3_signature_format(self, tmp_path):
        """New entries signed with v3:... format when HKDF enabled."""
        signer = CacheEntrySigner(
            key_file_path=tmp_path / "key.bin",
            use_hkdf_derivation=True,
        )
        sig = signer.sign_entry(self.SAMPLE_ENTRY)
        assert sig.startswith("v3:")
        assert len(sig) > 3  # "v3:" + hex

    def test_v2_entries_verify_after_hkdf_enabled(self, tmp_path):
        """Entries signed with v2 (HKDF off) still verify after HKDF is enabled."""
        key_file = tmp_path / "key.bin"

        # Sign with HKDF off (produces v2)
        signer_v2 = CacheEntrySigner(
            key_file_path=key_file,
            use_hkdf_derivation=False,
        )
        sig_v2 = signer_v2.sign_entry(self.SAMPLE_ENTRY)
        assert sig_v2.startswith("v2:")

        # Now create signer with HKDF on (same key file)
        signer_v3 = CacheEntrySigner(
            key_file_path=key_file,
            use_hkdf_derivation=True,
        )
        # v2 signature should still verify (uses master_key for v2)
        assert signer_v3.verify_entry(self.SAMPLE_ENTRY, sig_v2) is True

    def test_different_namespaces_different_signatures(self, tmp_path):
        """Same data, different namespace_id → different v3 signatures."""
        key_file = tmp_path / "key.bin"

        signer_a = CacheEntrySigner(
            key_file_path=key_file,
            namespace_id="namespace_a",
            use_hkdf_derivation=True,
        )
        signer_b = CacheEntrySigner(
            key_file_path=key_file,
            namespace_id="namespace_b",
            use_hkdf_derivation=True,
        )
        sig_a = signer_a.sign_entry(self.SAMPLE_ENTRY)
        sig_b = signer_b.sign_entry(self.SAMPLE_ENTRY)

        assert sig_a != sig_b
        assert sig_a.startswith("v3:")
        assert sig_b.startswith("v3:")

    def test_hkdf_disabled_uses_v2(self, tmp_path):
        """When use_hkdf_derivation=False, entries signed as v2."""
        signer = CacheEntrySigner(
            key_file_path=tmp_path / "key.bin",
            use_hkdf_derivation=False,
        )
        sig = signer.sign_entry(self.SAMPLE_ENTRY)
        assert sig.startswith("v2:")


# ---------------------------------------------------------------------------
# Namespace signature versioning tests
# ---------------------------------------------------------------------------


class TestNamespaceSignatureVersioning:
    """Tests for ns2 namespace signatures with HKDF-derived keys."""

    SAMPLE_NS = {
        "namespace_id": "test_ns",
        "display_name": "Test Namespace",
        "created_at": datetime(2026, 6, 15, tzinfo=timezone.utc),
    }

    def test_ns2_format_when_hkdf_enabled(self, tmp_path):
        """Namespace signed as ns2:{hex} when HKDF on."""
        signer = CacheEntrySigner(
            key_file_path=tmp_path / "key.bin",
            use_hkdf_derivation=True,
        )
        sig = signer.sign_namespace(self.SAMPLE_NS)
        assert sig.startswith("ns2:")
        assert signer.verify_namespace(self.SAMPLE_NS, sig) is True

    def test_ns1_entries_verify_after_hkdf_enabled(self, tmp_path):
        """Old ns1 signatures verify with master_key after HKDF enabled."""
        key_file = tmp_path / "key.bin"

        # Sign with HKDF off (produces ns1)
        signer_off = CacheEntrySigner(
            key_file_path=key_file,
            use_hkdf_derivation=False,
        )
        sig_ns1 = signer_off.sign_namespace(self.SAMPLE_NS)
        assert sig_ns1.startswith("ns1:")

        # Now create signer with HKDF on (same key file)
        signer_on = CacheEntrySigner(
            key_file_path=key_file,
            use_hkdf_derivation=True,
        )
        # ns1 signature should still verify (uses master_key for ns1)
        assert signer_on.verify_namespace(self.SAMPLE_NS, sig_ns1) is True

    def test_ns1_format_when_hkdf_disabled(self, tmp_path):
        """Namespace signed as ns1:{hex} when HKDF off."""
        signer = CacheEntrySigner(
            key_file_path=tmp_path / "key.bin",
            use_hkdf_derivation=False,
        )
        sig = signer.sign_namespace(self.SAMPLE_NS)
        assert sig.startswith("ns1:")
        assert signer.verify_namespace(self.SAMPLE_NS, sig) is True


# ---------------------------------------------------------------------------
# Config + factory integration tests
# ---------------------------------------------------------------------------


class TestHkdfConfigIntegration:
    """Tests for SecurityConfig and create_cache_signer factory."""

    def test_default_hkdf_enabled(self):
        """SecurityConfig().use_hkdf_derivation is True by default."""
        config = SecurityConfig()
        assert config.use_hkdf_derivation is True

    def test_hkdf_disabled_config(self):
        """SecurityConfig(use_hkdf_derivation=False) works."""
        config = SecurityConfig(use_hkdf_derivation=False)
        assert config.use_hkdf_derivation is False

    def test_create_cache_signer_passes_hkdf(self, tmp_path):
        """Factory passes namespace_id and use_hkdf_derivation to signer."""
        signer = create_cache_signer(
            cache_dir=tmp_path,
            namespace_id="my_namespace",
            use_hkdf_derivation=True,
        )
        assert signer.namespace_id == "my_namespace"
        assert signer.use_hkdf_derivation is True

        info = signer.get_field_info()
        assert info["namespace_id"] == "my_namespace"
        assert info["use_hkdf_derivation"] is True
